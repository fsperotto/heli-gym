"""Load the cockpit artwork with its original transparency."""

from pathlib import Path

import numpy as np
from PIL import Image


COCKPIT_IMAGE = Path(__file__).parent / "resources/figures/Cockpit_AW109_transparent.png"


def cockpit_rgba(path=COCKPIT_IMAGE):
    """Return packed RGBA pixels for the OpenGL texture."""
    with Image.open(path) as source:
        return np.array(source.convert("RGBA"))
