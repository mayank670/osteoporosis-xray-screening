"""Clinical risk factors from the dataset spreadsheet.

Only information a clinician could ask a patient BEFORE any bone-density
measurement is used. T-score, Z-score and Diagnosis are excluded on purpose:
they are (or are derived from) the bone-density measurement that defines the
label, so using them would be cheating ("target leakage").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Order matters: this is the order of the feature vector.
CLINICAL_FEATURES = (
    "age", "female", "height_m", "weight_kg", "bmi",
    "postmenopausal", "years_since_menopause",
    "smoker", "diabetic", "hypothyroidism", "estrogen_use",
    "fracture_history", "family_history", "walking_km",
)

_EXCLUDED_COLUMNS = ("T-score Value", "Z-Score Value", "Diagnosis")


def _yes(value) -> float:
    return 1.0 if str(value).strip().lower() == "yes" else 0.0


def _number(value, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def encode_clinical(age, sex, height_m, weight_kg, menopause_age=None, smoker=False, diabetic=False,
                    hypothyroidism=False, estrogen_use=False, fracture_history=False,
                    family_history=False, walking_km=None) -> np.ndarray:
    """One patient's answers -> feature vector in CLINICAL_FEATURES order.

    Used by both the training code and the web form, so they cannot disagree.
    """
    age = float(age)
    height_m, weight_kg = float(height_m), float(weight_kg)
    female = 1.0 if str(sex).strip().lower() in ("female", "f") else 0.0
    menopause = _number(menopause_age)
    postmenopausal = 1.0 if female and not np.isnan(menopause) and menopause > 0 else 0.0
    years_since = max(0.0, age - menopause) if postmenopausal else 0.0
    walking = _number(walking_km, default=1.0)  # median-ish default when unknown
    return np.array([
        age, female, height_m, weight_kg, weight_kg / (height_m ** 2),
        postmenopausal, years_since,
        float(bool(smoker)), float(bool(diabetic)), float(bool(hypothyroidism)), float(bool(estrogen_use)),
        float(bool(fracture_history)), float(bool(family_history)), walking,
    ], dtype=np.float64)


def load_clinical_table(dataset_dir: Path) -> dict[str, np.ndarray]:
    """{patient_id: feature vector} for every row of the dataset spreadsheet."""
    import openpyxl

    workbook_path = next(Path(dataset_dir).glob("*.xlsx"))
    sheet = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True).worksheets[0]
    rows = list(sheet.iter_rows(values_only=True))
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    col = {name: i for i, name in enumerate(header)}

    def get(row, prefix):
        index = next(i for name, i in col.items() if name.lower().startswith(prefix.lower()))
        return row[index]

    table = {}
    for row in rows[1:]:
        patient = row[col["Patient Id"]]
        if not patient:
            continue
        fracture = str(get(row, "History of Fracture") or "no").strip().lower()
        table[str(patient).strip().upper()] = encode_clinical(
            age=get(row, "Age"), sex=get(row, "Gender"), height_m=get(row, "height"),
            weight_kg=get(row, "Weight"), menopause_age=get(row, "Menopause Age"),
            smoker=_yes(get(row, "Smoker")), diabetic=_yes(get(row, "Diabetic")),
            hypothyroidism=_yes(get(row, "Hypothyroidism")), estrogen_use=_yes(get(row, "Estrogen Use")),
            fracture_history=fracture not in ("no", "none", ""), family_history=_yes(get(row, "Family History")),
            walking_km=get(row, "Maximum Walking"),
        )
    return table
