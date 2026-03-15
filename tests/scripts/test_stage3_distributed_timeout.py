from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from accelerate.utils import InitProcessGroupKwargs
from hydra import compose, initialize_config_dir

from tests.helpers.paths import CONFIG_DIR, REPO_ROOT, script_path


def _load_script_module(name: str, module_name: str):
    path = script_path(name)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load script module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage3_train_config_writes_distributed_timeout(tmp_path: Path) -> None:
    mod = _load_script_module("6_train_lerobot.py", "stage3_train_script_timeout")
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=stage3_local",
                "lerobot.distributed_timeout_s=1800",
            ],
        )

    config_path = mod._write_lerobot_train_config(cfg, runtime_cwd=tmp_path)
    payload = json.loads(config_path.read_text())
    assert payload["distributed_timeout_s"] == 1800.0


def test_stage3_train_config_writes_resume_fields(tmp_path: Path) -> None:
    mod = _load_script_module("6_train_lerobot.py", "stage3_train_script_resume")
    checkpoint_dir = tmp_path / "020000" / "pretrained_model"
    checkpoint_dir.mkdir(parents=True)
    resume_config = checkpoint_dir / "train_config.json"
    resume_config.write_text("{}\n")

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=stage3_local",
                "lerobot.resume=true",
                f"lerobot.resume_from={resume_config}",
            ],
        )

    config_path = mod._write_lerobot_train_config(cfg, runtime_cwd=tmp_path)
    payload = json.loads(config_path.read_text())
    assert payload["resume"] is True
    assert payload["resume_config_path"] == str(resume_config.resolve())


def test_accelerator_kwargs_handlers_include_init_pg_timeout() -> None:
    lerobot_src = REPO_ROOT / "lerobot" / "src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))

    from lerobot.scripts.lerobot_train import _accelerator_kwargs_handlers

    cfg = SimpleNamespace(distributed_timeout_s=1800)
    handlers = _accelerator_kwargs_handlers(cfg)

    init_pg_handler = next(
        handler for handler in handlers if isinstance(handler, InitProcessGroupKwargs)
    )
    assert init_pg_handler.timeout.total_seconds() == 1800
