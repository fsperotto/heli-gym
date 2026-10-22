# Native renderer binding audit

Scope: `renderer/pyapi.py`, `renderer/api.py`, the exported C interface, and
native ownership of data/handles crossing that interface.

| Finding | Correction |
| --- | --- |
| GUI stored pointers into temporary NumPy arrays and Python strings | Native GUI owns copies of labels and values. |
| Several C `void` returns declared as pointers, and two `restypes` typos | Correct ctypes `restype=None`; return/argument declarations checked against the C++ header. |
| Window dimensions declared signed although the C interface accepts unsigned | Matching `c_uint` declarations plus positive, bounded dimension validation. |
| GUI buffers could be shorter than the stored row count | Checked C exports verify section IDs and row counts before copying; Python checks shape, length, float32 type, alignment and contiguity. |
| GUI registration mutated the caller's labels into bytes | Encode into a separate array, leaving labels unchanged. |
| Arbitrary printf placeholders could consume nonexistent/wrongly typed arguments | Labels permit at most one floating-point conversion per row, plus escaped percent signs. |
| Camera getter exposed a borrowed native float pointer | Return a copied three-value NumPy snapshot; high-level getter converts it to NED, matching the setter. |
| Scaling negated an axis, reflecting models | Scaling permutes axes without changing their signs. |
| Graphics/model construction failures could leave null or unusable objects | Native construction errors return null handles; Python raises a useful error. |
| Invalid FPS could divide by zero or overflow the native timer; Windows `long` could truncate long intervals | Validate FPS and use the nanosecond representation type. |
| Most renderer methods could run after close/termination | Reject accesses through the high-level renderer after it closes. Existing idempotent close/render/wait operations remain safe. |
| Window/model/camera/GUI allocations had no native owner | Track windows/models with owning pointers; termination releases their allocations. GUI shutdown is idempotent. |
| Global GLFW/ImGui state was unsafe for concurrent renderer windows | Reject a second active window explicitly; terminate the first before creating another. |

Validation:

- Python ABI tests compare exported argument and return types against `py_api.h`.
- Python tests cover GUI validation, string ownership, copied camera snapshots,
  coordinate round trips, invalid handles/FPS, and lifecycle checks.
- Native headless tests exercise GUI copies after source buffers are modified or
  leave scope, invalid section/length rejection, camera pointers, FPS, and keys.
- Real ctypes calls into the rebuilt shared library reject invalid GUI handles
  without opening a graphics window.
- Full Python suite passed (105 tests); final binding/lifecycle tests passed.
- AddressSanitizer headless checks run with leak detection disabled. LeakSanitizer
  cannot run in this sandbox because it is traced (ptrace).

Limits: this is a single-window, single graphics-thread interface. Raw native
handles are borrowed and must not be reused after termination. Headless checks do
not validate visual output, GPU-driver behavior, or Windows DLL execution. A
Windows renderer rebuild and visual GUI session remain manual checks. This audit
does not certify the entire rendering engine free of errors.

The updated bindings require rebuilding the native renderer with
`bash compile_renderer.sh` on each machine where rendering runs, and restarting
Python so it loads the new library.
