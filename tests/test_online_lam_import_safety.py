from __future__ import annotations

import builtins
import importlib
import sys


def test_online_lam_import_does_not_require_lam_inference(monkeypatch):
    sys.modules.pop("stage2.online_lam", None)

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "lam.inference":
            raise ModuleNotFoundError("No module named 'lam.inference'")
        if name == "lam" and fromlist and "inference" in fromlist:
            raise ModuleNotFoundError("No module named 'lam.inference'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = importlib.import_module("stage2.online_lam")
    assert hasattr(module, "frames_to_lam_video")
    assert hasattr(module, "LAMTaskCodeProvider")
