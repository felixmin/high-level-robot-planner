from __future__ import annotations

from common.hydra_overrides import autoquote_override_value, normalize_overrides


def test_autoquote_override_value_no_change_without_equals():
    assert autoquote_override_value("experiment=vla") == "experiment=vla"
    assert autoquote_override_value("seed=123") == "seed=123"


def test_autoquote_override_value_quotes_value_containing_equals():
    ov = "training.resume_from_checkpoint=/tmp/vla-stepstep=079000.ckpt"
    out = autoquote_override_value(ov)
    assert out.startswith("training.resume_from_checkpoint=")
    assert out.endswith('"/tmp/vla-stepstep=079000.ckpt"')


def test_autoquote_override_value_leaves_already_quoted():
    ov = 'training.resume_from_checkpoint="/tmp/vla-stepstep=079000.ckpt"'
    assert autoquote_override_value(ov) == ov


def test_normalize_overrides_is_elementwise():
    ovs = [
        "experiment=vla",
        "training.resume_from_checkpoint=/tmp/vla-stepstep=079000.ckpt",
        "seed=0",
    ]
    out = normalize_overrides(ovs)
    assert out[0] == "experiment=vla"
    assert out[1].startswith("training.resume_from_checkpoint=")
    assert out[1].endswith('"/tmp/vla-stepstep=079000.ckpt"')
    assert out[2] == "seed=0"

