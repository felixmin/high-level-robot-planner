from common.adapters.oxe_shared import (
    resolve_oxe_dataset_config,
    resolve_oxe_dataset_key,
)


def test_resolve_robonet_alias_from_folder_name() -> None:
    assert resolve_oxe_dataset_key("robo_net") == "robonet"
    cfg = resolve_oxe_dataset_config("robo_net")
    assert cfg is not None
    assert cfg.name == "robonet"


def test_resolve_rt1_alias_from_folder_name() -> None:
    assert resolve_oxe_dataset_key("fractal20220817_data") == "rt1"
    cfg = resolve_oxe_dataset_config("fractal20220817_data")
    assert cfg is not None
    assert cfg.name == "rt1"


def test_resolve_alias_normalization() -> None:
    assert resolve_oxe_dataset_key("ROBO-NET") == "robonet"


def test_unknown_dataset_returns_none() -> None:
    assert resolve_oxe_dataset_key("totally_unknown_dataset") is None
    assert resolve_oxe_dataset_config("totally_unknown_dataset") is None


def test_octo_dataset_entries_resolve() -> None:
    names = [
        "austin_buds_dataset_converted_externally_to_rlds",
        "austin_sailor_dataset_converted_externally_to_rlds",
        "austin_sirius_dataset_converted_externally_to_rlds",
        "cmu_stretch",
        "dlr_edan_shared_control_converted_externally_to_rlds",
        "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
        "nyu_door_opening_surprising_effectiveness",
        "nyu_franka_play_dataset_converted_externally_to_rlds",
    ]
    for name in names:
        cfg = resolve_oxe_dataset_config(name)
        assert cfg is not None, name
