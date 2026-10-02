import sys
import types
from .cellcast import *  # noqa: F403
from . import cellcast as _ext

# HACK: Given the current crate setup, maturin installs the Rust extension as
# `cellcast.cellcast`, so PyO3 registers submodules (e.g. `models`) under
# `cellcast.cellcast.*` in sys.modules rather than `cellcast.*` This
# __init__.py brings all of them in and adds them as `cellcast.*` instead.
# An alternative design would be to have separate crates per submodule - maybe
# that would make sense down the line.
# NOTE that `cellcast.cellcast` will still be present in `sys.modules`, but
# that's okay for now.
for _name in dir(_ext):
    _obj = getattr(_ext, _name)
    if isinstance(_obj, types.ModuleType):
        sys.modules.setdefault(f"cellcast.{_name}", _obj)
