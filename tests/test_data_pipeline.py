import numpy as np
import pytest

from config import CLASS_NAMES, PREPROCESSING
from ml.data.loader import class_weights, load_arrays
from ml.data.manifest import build_manifest, load_manifest, save_manifest
from ml.data.splits import assign_parts, check_no_leakage, make_fold


@pytest.fixture
def manifest(synthetic_dataset):
    return build_manifest(synthetic_dataset)


def by_id(records):
    return {r.patient_id: r for r in records}


def test_unreadable_file_is_excluded_with_reason(manifest):
    record = by_id(manifest)["N99"]
    assert record.excluded and record.reason.startswith("unreadable")


def test_same_class_duplicate_keeps_lowest_id(manifest):
    records = by_id(manifest)
    assert not records["OP1"].excluded
    assert records["OP11"].excluded and "exact duplicate of OP1" in records["OP11"].reason


def test_conflicting_labels_exclude_every_copy(manifest):
    records = by_id(manifest)
    assert records["N1"].excluded and records["OS11"].excluded
    assert "conflicting labels" in records["N1"].reason


def test_near_duplicate_shares_group(manifest):
    records = by_id(manifest)
    assert "OP2" in records["OP12"].near_duplicates
    assert records["OP12"].group == records["OP2"].group


def test_distinct_images_get_own_groups(manifest):
    records = by_id(manifest)
    assert records["OP3"].group != records["OP4"].group


def test_manifest_csv_round_trip(manifest, tmp_path):
    save_manifest(manifest, tmp_path / "m.csv")
    loaded = load_manifest(tmp_path / "m.csv")
    assert [(r.patient_id, r.excluded, r.group, r.near_duplicates) for r in loaded] == \
           [(r.patient_id, r.excluded, r.group, r.near_duplicates) for r in manifest]


def test_missing_class_folder_raises(tmp_path):
    (tmp_path / "normal").mkdir()
    with pytest.raises(FileNotFoundError):
        build_manifest(tmp_path)


def test_folds_have_no_leakage_and_cover_everything(manifest):
    parts = assign_parts(manifest, n_folds=3, seed=1)
    usable = [r for r in manifest if not r.excluded]
    tested = []
    for fold in range(3):
        split = make_fold(manifest, parts, fold, 3)
        check_no_leakage(split)
        assert split.train and split.val and split.test
        tested += [r.patient_id for r in split.test]
    assert sorted(tested) == sorted(r.patient_id for r in usable)  # each image tested exactly once
    # the near-duplicate pair always lands in the same part
    assert parts["OP2"] == parts["OP12"]


def test_folds_are_reproducible(manifest):
    assert assign_parts(manifest, 3, seed=7) == assign_parts(manifest, 3, seed=7)


def test_leakage_check_detects_shared_patient(manifest):
    parts = assign_parts(manifest, 3, seed=1)
    split = make_fold(manifest, parts, 0, 3)
    split.test.append(split.train[0])
    with pytest.raises(AssertionError, match="Leakage"):
        check_no_leakage(split)


def test_load_arrays_contract_and_cache(manifest, synthetic_dataset, tmp_path):
    usable = [r for r in manifest if not r.excluded][:5]
    X, y = load_arrays(usable, synthetic_dataset, PREPROCESSING, CLASS_NAMES, tmp_path / "cache")
    assert X.shape == (5, 224, 224, 3) and X.dtype == np.float32 and y.dtype == np.int32
    assert len(list((tmp_path / "cache").glob("*.npy"))) == 1
    X2, _ = load_arrays(usable, synthetic_dataset, PREPROCESSING, CLASS_NAMES, tmp_path / "cache")
    assert np.array_equal(X, X2)


def test_class_weights_balanced_formula():
    y = np.array([0] * 18 + [1] * 82 + [2] * 26)
    weights = class_weights(y, 3)
    assert weights[0] == pytest.approx(126 / (3 * 18))
    assert weights[1] < 1 < weights[2] < weights[0]


def test_class_weights_reject_missing_class():
    with pytest.raises(ValueError, match="missing"):
        class_weights(np.array([0, 0, 1]), 3)


def test_augmentation_only_when_training():
    from ml.data.augmentation import build_augmenter
    from ml.data.loader import make_dataset

    X = np.random.default_rng(0).uniform(0, 255, (4, 64, 64, 3)).astype(np.float32)
    y = np.arange(4) % 3
    augmenter = build_augmenter(seed=0)
    eval_batch = next(iter(make_dataset(X, y, 4, training=False, seed=0, augmenter=augmenter)))[0].numpy()
    assert np.array_equal(eval_batch, X)  # untouched, same order
    train_batch = next(iter(make_dataset(X, y, 4, training=True, seed=0, augmenter=augmenter)))[0].numpy()
    assert not np.array_equal(np.sort(train_batch.ravel()), np.sort(X.ravel()))
