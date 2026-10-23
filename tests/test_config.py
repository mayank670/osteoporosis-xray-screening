import dataclasses
from pathlib import Path

import pytest

import config


def test_three_classes_in_fixed_order():
    assert config.CLASS_NAMES == ("Normal", "Osteopenia", "Osteoporosis")


def test_relative_env_path_is_resolved_from_project_root(monkeypatch):
    monkeypatch.setenv("SOME_TEST_DIR", "my/data")
    assert config.path_from_env("SOME_TEST_DIR", "unused") == config.PROJECT_ROOT / "my" / "data"


def test_absolute_env_path_is_kept(monkeypatch, tmp_path):
    monkeypatch.setenv("SOME_TEST_DIR", str(tmp_path))
    assert config.path_from_env("SOME_TEST_DIR", "unused") == tmp_path


def test_default_used_when_env_var_missing(monkeypatch):
    monkeypatch.delenv("SOME_TEST_DIR", raising=False)
    assert config.path_from_env("SOME_TEST_DIR", "fallback") == config.PROJECT_ROOT / "fallback"


def test_preprocessing_config_is_read_only():
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.PREPROCESSING.use_clahe = True


def test_preprocessing_config_serialises_to_dict():
    settings = config.PREPROCESSING.to_dict()
    assert settings["image_size"] == (224, 224)
    assert settings["version"] == "prep-v1"
    assert isinstance(config.DATASET_DIR, Path)
