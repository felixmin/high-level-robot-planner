from __future__ import annotations

from types import SimpleNamespace

from common.callbacks import DatasetUsageLoggerCallback


def test_dataset_usage_logger_counts_and_prints_on_validation(capsys):
    cb = DatasetUsageLoggerCallback(enabled=True, log_on_validation_end=True)
    trainer = SimpleNamespace(global_step=7)

    cb.on_train_batch_end(
        trainer,
        None,
        outputs=None,
        batch={"dataset_name": ["bridge", "bridge", "robonet"]},
        batch_idx=0,
    )

    cb.on_validation_end(trainer, None)
    out = capsys.readouterr().out
    assert "[Validation][DatasetUsage]" in out
    assert "bridge=2" in out
    assert "robonet=1" in out

    # Second validation should not re-print if no new data arrived.
    cb.on_validation_end(trainer, None)
    out2 = capsys.readouterr().out
    assert out2 == ""


def test_dataset_usage_logger_can_log_every_n_steps(capsys):
    cb = DatasetUsageLoggerCallback(enabled=True, log_on_validation_end=False, log_every_n_steps=2)
    trainer = SimpleNamespace(global_step=0)

    cb.on_train_batch_end(
        trainer,
        None,
        outputs=None,
        batch={"dataset_name": ["bridge"]},
        batch_idx=0,
    )
    assert capsys.readouterr().out == ""

    trainer.global_step = 1
    cb.on_train_batch_end(
        trainer,
        None,
        outputs=None,
        batch={"dataset_name": ["bridge", "robonet"]},
        batch_idx=0,
    )
    out = capsys.readouterr().out
    assert "[Train][DatasetUsage]" in out
    assert "interval_total=3" in out


def test_dataset_usage_logger_can_print_batch_mix(capsys):
    cb = DatasetUsageLoggerCallback(
        enabled=True,
        log_on_validation_end=False,
        log_batch_composition_every_n_steps=2,
    )
    trainer = SimpleNamespace(global_step=0)
    cb.on_train_batch_end(
        trainer,
        None,
        outputs=None,
        batch={"dataset_name": ["bridge", "bridge", "robonet"]},
        batch_idx=0,
    )
    assert capsys.readouterr().out == ""

    trainer.global_step = 1
    cb.on_train_batch_end(
        trainer,
        None,
        outputs=None,
        batch={"dataset_name": ["bridge", "language_table", "language_table"]},
        batch_idx=0,
    )
    out = capsys.readouterr().out
    assert "[Train][BatchMix]" in out
    assert "batch_total=3" in out
