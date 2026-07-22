import builtins
import importlib
import sys


def test_launcher_imports_without_unix_pty_modules(monkeypatch):
    real_import = builtins.__import__

    def block_unix_pty_modules(name, *args, **kwargs):
        if name in {"pty", "termios"}:
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    original = sys.modules.pop("sflow.core.launcher", None)
    package = importlib.import_module("sflow.core")
    original_package_attr = getattr(package, "launcher", None)
    monkeypatch.setattr(builtins, "__import__", block_unix_pty_modules)

    try:
        launcher = importlib.import_module("sflow.core.launcher")
        assert launcher.pty is None
        importlib.import_module("sflow.core.orchestrator")
    finally:
        sys.modules.pop("sflow.core.launcher", None)
        if original is not None:
            sys.modules["sflow.core.launcher"] = original
            setattr(package, "launcher", original_package_attr)
