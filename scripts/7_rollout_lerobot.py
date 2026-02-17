#!/usr/bin/env python3
"""
Script 7: Stage-3 LeRobot rollout/evaluation entrypoint.

This script is intended to run inside the Slurm container launched by
`scripts/submit_job.py`. It can optionally editable-install a local LeRobot
policy plugin and then execute `lerobot-eval`.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "packages"))

from common.cache_env import configure_cache_env, resolve_cache_dir
from common.unified_logging import resolve_runs_dir, setup_unified_logging


def _to_bool_flag(value: object) -> str:
    return "true" if bool(value) else "false"


def _run_install_command(
    cmd: list[str],
    *,
    logger,
    cwd: Path,
    env: dict[str, str],
) -> tuple[bool, str]:
    logger.info("Install command: %s", shlex.join(cmd))
    try:
        subprocess.run(cmd, cwd=str(cwd), env=env, check=True)
        return True, ""
    except subprocess.CalledProcessError as e:
        err = f"Command failed ({e.returncode}): {shlex.join(cmd)}"
        logger.warning(err)
        return False, err


def _install_editable_policy(
    *,
    editable_path: Path,
    logger,
    cwd: Path,
    env: dict[str, str],
) -> None:
    python = sys.executable
    attempted_errors: list[str] = []

    ok, err = _run_install_command(
        [python, "-m", "pip", "install", "--no-deps", "-e", str(editable_path)],
        logger=logger,
        cwd=cwd,
        env=env,
    )
    if ok:
        return
    attempted_errors.append(err)

    if "No module named pip" in err:
        logger.info("pip module missing; attempting bootstrap via ensurepip")
        boot_ok, boot_err = _run_install_command(
            [python, "-m", "ensurepip", "--upgrade"],
            logger=logger,
            cwd=cwd,
            env=env,
        )
        if boot_ok:
            ok, err = _run_install_command(
                [python, "-m", "pip", "install", "--no-deps", "-e", str(editable_path)],
                logger=logger,
                cwd=cwd,
                env=env,
            )
            if ok:
                return
            attempted_errors.append(err)
        else:
            attempted_errors.append(boot_err)

    uv_bin = shutil.which("uv", path=env.get("PATH"))
    if uv_bin is not None:
        ok, err = _run_install_command(
            [
                uv_bin,
                "pip",
                "install",
                "--python",
                python,
                "--no-deps",
                "-e",
                str(editable_path),
            ],
            logger=logger,
            cwd=cwd,
            env=env,
        )
        if ok:
            return
        attempted_errors.append(err)
    else:
        attempted_errors.append("uv executable not found on PATH")

    pip_bin = shutil.which("pip", path=env.get("PATH"))
    if pip_bin is not None:
        ok, err = _run_install_command(
            [pip_bin, "install", "--no-deps", "-e", str(editable_path)],
            logger=logger,
            cwd=cwd,
            env=env,
        )
        if ok:
            return
        attempted_errors.append(err)
    else:
        attempted_errors.append("pip executable not found on PATH")

    details = "\n".join(f"- {msg}" for msg in attempted_errors)
    raise RuntimeError(
        "Failed to editable-install policy package with all supported installers:\n"
        f"{details}"
    )


def _command_from_cfg(cfg: DictConfig) -> list[str]:
    eval_cmd = str(OmegaConf.select(cfg, "lerobot_eval.command") or "lerobot-eval")

    policy_path = OmegaConf.select(cfg, "lerobot_eval.policy_path")
    env_type = OmegaConf.select(cfg, "lerobot_eval.env_type")
    n_episodes = OmegaConf.select(cfg, "lerobot_eval.eval_n_episodes")
    batch_size = OmegaConf.select(cfg, "lerobot_eval.eval_batch_size")

    required = {
        "lerobot_eval.policy_path": policy_path,
        "lerobot_eval.env_type": env_type,
        "lerobot_eval.eval_n_episodes": n_episodes,
        "lerobot_eval.eval_batch_size": batch_size,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Missing required lerobot_eval config keys: {missing}")

    cmd = [
        eval_cmd,
        f"--policy.path={policy_path}",
        f"--env.type={env_type}",
        f"--eval.n_episodes={int(n_episodes)}",
        f"--eval.batch_size={int(batch_size)}",
    ]

    env_task = OmegaConf.select(cfg, "lerobot_eval.env_task")
    if env_task:
        cmd.append(f"--env.task={env_task}")

    policy_device = OmegaConf.select(cfg, "lerobot_eval.policy_device")
    if policy_device is not None:
        cmd.append(f"--policy.device={policy_device}")

    policy_use_amp = OmegaConf.select(cfg, "lerobot_eval.policy_use_amp")
    if policy_use_amp is not None:
        cmd.append(f"--policy.use_amp={_to_bool_flag(policy_use_amp)}")

    output_dir = OmegaConf.select(cfg, "lerobot_eval.output_dir")
    if output_dir:
        cmd.append(f"--output_dir={output_dir}")

    job_name = OmegaConf.select(cfg, "lerobot_eval.job_name")
    if job_name:
        cmd.append(f"--job_name={job_name}")

    seed = OmegaConf.select(cfg, "lerobot_eval.seed")
    if seed is not None:
        cmd.append(f"--seed={int(seed)}")

    trust_remote_code = OmegaConf.select(cfg, "lerobot_eval.trust_remote_code")
    if trust_remote_code is not None:
        cmd.append(f"--trust_remote_code={_to_bool_flag(trust_remote_code)}")

    extra_args = OmegaConf.select(cfg, "lerobot_eval.extra_args") or []
    if not (OmegaConf.is_list(extra_args) or isinstance(extra_args, (list, tuple))):
        raise ValueError("lerobot_eval.extra_args must be a list of strings")
    for i, arg in enumerate(extra_args):
        if not isinstance(arg, str):
            raise ValueError(f"lerobot_eval.extra_args[{i}] must be a string")
        if arg.strip():
            cmd.append(arg.strip())

    return cmd


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    runs_dir = None
    try:
        if HydraConfig.initialized():
            runs_dir = Path(str(HydraConfig.get().runtime.output_dir))
    except Exception:
        runs_dir = None
    if runs_dir is None:
        runs_dir = resolve_runs_dir(
            logging_root_dir=OmegaConf.select(cfg, "logging.root_dir"),
            logging_runs_dir=OmegaConf.select(cfg, "logging.runs_dir"),
            workspace_root=workspace_root,
            experiment_name=OmegaConf.select(cfg, "experiment.name"),
        )

    logger, _ = setup_unified_logging(
        runs_dir=runs_dir,
        job_id=OmegaConf.select(cfg, "logging.job_id"),
        log_level=str(OmegaConf.select(cfg, "logging.level") or "INFO"),
        logger_name="lerobot.rollout",
    )

    logger.info("=" * 80)
    logger.info("LAPA Stage 3: LeRobot Rollout")
    logger.info("=" * 80)
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))

    cache_dir = resolve_cache_dir(cfg=cfg, workspace_root=workspace_root)
    if cache_dir is not None:
        configure_cache_env(cache_dir=cache_dir, logger=logger)

    env = os.environ.copy()
    packages_path = str(workspace_root / "packages")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        packages_path if not existing_pythonpath else f"{packages_path}:{existing_pythonpath}"
    )
    env_overrides = OmegaConf.select(cfg, "lerobot_eval.env") or {}
    if not OmegaConf.is_dict(env_overrides):
        raise ValueError("lerobot_eval.env must be a mapping of environment variables")
    for k, v in env_overrides.items():
        if v is None:
            continue
        env[str(k)] = str(v)

    install_editable = OmegaConf.select(cfg, "lerobot_eval.install_policy_editable")
    if install_editable:
        editable_path = Path(str(install_editable))
        if not editable_path.is_absolute():
            editable_path = workspace_root / editable_path
        if not editable_path.exists():
            raise FileNotFoundError(f"Editable policy path not found: {editable_path}")
        logger.info("Installing editable policy package: %s", editable_path)
        _install_editable_policy(
            editable_path=editable_path,
            logger=logger,
            cwd=workspace_root,
            env=env,
        )

    cmd = _command_from_cfg(cfg)
    logger.info("Launching LeRobot rollout command:")
    logger.info("  %s", shlex.join(cmd))
    subprocess.run(cmd, cwd=str(workspace_root), env=env, check=True)
    logger.info("Stage 3 rollout complete.")


if __name__ == "__main__":
    main()
