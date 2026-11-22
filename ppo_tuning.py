"""Search PPO settings, then retrain finalists and the default PPO baseline."""

import json
from pathlib import Path
import statistics
import subprocess
import sys

import main as runner


ENVIRONMENT_FIELDS = (
    "task", "heli_name", "episode_seconds", "flip_angle", "continual", "random_wind",
    "wind_speed", "wind_direction", "turbulence_level", "smoothness_weight",
    *runner.TOLERANCE_OPTIONS,
)


def run_candidate(args, directory, params, train_seed, eval_seed, timesteps):
    common = [sys.executable, str(Path(runner.__file__).resolve()), "--algorithm", "ppo",
              "--device", args.device, "--target-dir", str(directory),
              "--n-envs", str(args.n_envs), "--torch-threads", str(args.torch_threads)]
    train = [*common, "train", "--no-tensorboard", "--total-timesteps", str(timesteps),
             "--seed", str(train_seed), "--no-wind" if args.no_wind else "--wind"]
    for name in ENVIRONMENT_FIELDS:
        value = getattr(args, name)
        if isinstance(value, bool):
            if value:
                train.append(f"--{name.replace('_', '-')}")
        elif value is not None:
            train.extend([f"--{name.replace('_', '-')}", str(value)])
    for name, value in params.items():
        train.extend([f"--{name.replace('_', '-')}", str(value)])
    subprocess.run(train, check=True)

    # main.py offsets evaluation seeds by n_envs and samples wind per episode.
    subprocess.run([*common, "eval", "--seed", str(eval_seed - args.n_envs),
                    "--eval-episodes", str(args.eval_episodes)], check=True)
    return json.loads((directory / "evaluation.json").read_text())["mean_reward"]


def suggest(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 3e-5, 1e-3, log=True),
        "gamma": trial.suggest_categorical("gamma", [0.99, 0.995, 0.997, 0.999]),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.90, 0.99),
        "clip_range": trial.suggest_float("clip_range", 0.10, 0.30),
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048]),
    }


def export_final_params(root, args, winner):
    defaults = runner.parse_args(["train", "--algorithm", "ppo"])
    ppo = {name: getattr(defaults, name) for name in (
        "learning_rate", "gamma", "gae_lambda", "clip_range", "n_epochs",
        "ent_coef", "vf_coef", "max_grad_norm", "net_arch")}
    ppo.update(n_steps=1024, batch_size=256, n_envs=args.n_envs)
    ppo.update(winner["params"])
    runner.write_json(root / "final_params.json", {
        **{key: getattr(args, key) for key in ENVIRONMENT_FIELDS},
        "algorithm": "ppo", "no_wind": args.no_wind, **ppo,
    })


def print_summary(path):
    if path.is_file():
        print(path.read_text(), flush=True)
    else:
        print(f"No saved tuning summary in {path.parent}", flush=True)


def tune(args):
    root = (args.target_dir / "optuna").resolve()
    summary_path = root / "summary.json"
    if root.exists() or args.summary_only:
        print_summary(summary_path)
    if args.summary_only:
        return

    try:
        import optuna
    except ModuleNotFoundError as error:
        if error.name != "optuna":
            raise
        raise SystemExit("Install Optuna in this environment with python -m pip install optuna") from error

    settings = {key: getattr(args, key) for key in (*ENVIRONMENT_FIELDS, "n_envs",
                "torch_threads", "device", "seed", "eval_episodes", "refinement_seeds",
                "trial_timesteps", "refinement_timesteps")}
    settings["wind"] = not args.no_wind
    config_path = root / "tuning.json"
    if config_path.exists():
        saved_settings = json.loads(config_path.read_text())
        saved_settings.setdefault("continual", args.task == "goto")
        for key, value in dict(heli_name="aw109", episode_seconds=40., flip_angle=None,
                               wind_direction=None, n_envs=4, torch_threads=1).items():
            saved_settings.setdefault(key, value)
        if saved_settings != settings:
            raise SystemExit("Tuning settings differ from the saved study. Use its settings or choose another --target-dir.")
    else:
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise SystemExit("Target directory contains files from another run. Choose another --target-dir.")
        runner.write_json(config_path, settings)

    search_eval_seed = max(args.n_envs, args.seed + args.refinement_seeds + 1)
    refinement_eval_seed = search_eval_seed + args.eval_episodes
    if refinement_eval_seed + args.eval_episodes > 2**32:
        raise SystemExit("Seed and episode counts exceed the simulator's seed range.")
    previous_refinement = (json.loads(summary_path.read_text()).get("refinement", [])
                           if summary_path.is_file() else [])

    existing_runs = len(list((root / "search").glob("trial-*")))
    study = optuna.create_study(
        study_name=args.task, storage=f"sqlite:///{root / 'study.db'}",
        direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=(args.seed + existing_runs) % 2**32),
    )

    def save_summary():
        completed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        ranked = sorted(completed, key=lambda trial: (-trial.value, trial.number))
        summary = {
            "settings": settings,
            "search": [{"trial": trial.number, "mean_reward": trial.value, "params": trial.params}
                       for trial in ranked],
            "refinement": previous_refinement,
        }
        runner.write_json(summary_path, summary)
        return summary

    def objective(trial):
        return run_candidate(args, root / "search" / f"trial-{trial.number:04d}",
                             suggest(trial), args.seed, search_eval_seed, args.trial_timesteps)

    study.optimize(objective, n_trials=args.n_trials,
                   callbacks=[lambda _study, _trial: save_summary()])
    summary = save_summary()
    if not summary["search"]:
        print_summary(summary_path)
        return

    candidates = [("ppo-defaults", {})]
    candidates.extend((f"trial-{row['trial']:04d}", row["params"])
                      for row in summary["search"][:args.finalists])
    summary["refinement"] = []
    for name, params in candidates:
        scores = []
        for index in range(args.refinement_seeds):
            seed = args.seed + index + 1
            directory = root / "refinement" / name / f"seed-{seed}"
            result_path = directory / "refinement_result.json"
            if result_path.is_file():
                score = json.loads(result_path.read_text())["mean_reward"]
            else:
                score = run_candidate(args, directory, params, seed, refinement_eval_seed,
                                      args.refinement_timesteps)
                runner.write_json(result_path, {"mean_reward": score})
            scores.append({"seed": seed, "mean_reward": score})
        summary["refinement"].append({
            "candidate": name, "params": params, "seeds": scores,
            "mean_reward": statistics.mean(row["mean_reward"] for row in scores),
            "std_reward": statistics.pstdev(row["mean_reward"] for row in scores),
        })
        summary["refinement"].sort(key=lambda row: (-row["mean_reward"], row["candidate"]))
        runner.write_json(summary_path, summary)
    export_final_params(root, args, summary["refinement"][0])
    print_summary(summary_path)

