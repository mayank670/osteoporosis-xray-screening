"""Web application tests. The model is the tiny UNTRAINED test model: these
tests check software behaviour, never medical performance."""

import re

import numpy as np
import pytest

from app import create_app
from app.extensions import db
from app.models import Prediction, Report, Scan, User
from tests.conftest import encode, make_xray_like

PASSWORD = "correct-horse-1"


@pytest.fixture
def app(tmp_path, tiny_model_dir):
    app = create_app("testing", {
        "UPLOADS_DIR": tmp_path / "uploads", "REPORTS_DIR": tmp_path / "reports",
        "LOG_DIR": tmp_path / "logs", "MODEL_DIR": tiny_model_dir, "MAX_CONTENT_LENGTH": 2 * 1024 * 1024,
    })
    yield app
    with app.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


def register(client, email="alice@example.com", name="Alice", password=PASSWORD):
    return client.post("/register", data={"name": name, "email": email, "password": password, "confirm": password})


def login(client, email="alice@example.com", password=PASSWORD):
    return client.post("/login", data={"email": email, "password": password})


def upload(client, data: bytes, filename="knee.png", accept_json=True):
    headers = {"Accept": "application/json"} if accept_json else {}
    return client.post("/predict", data={"image": (__import__("io").BytesIO(data), filename)},
                       content_type="multipart/form-data", headers=headers)


@pytest.fixture
def alice(client):
    register(client)
    login(client)
    return client


@pytest.fixture
def png():
    return encode(make_xray_like(3, (320, 280)))

# ---------- health ----------

def test_health_reports_model(client):
    body = client.get("/health").get_json()
    assert body == {"status": "ok", "model_loaded": True, "model_version": "tiny-test-model",
                    "hybrid_available": True}

# ---------- registration and login ----------

def test_register_and_login_flow(client, app):
    assert register(client).status_code == 302
    with app.app_context():
        user = User.query.filter_by(email="alice@example.com").one()
        assert user.password_hash != PASSWORD and user.check_password(PASSWORD)
        assert user.role == "user"
    assert login(client).status_code == 302
    assert client.get("/upload").status_code == 200


def test_email_is_normalised_and_unique(client):
    register(client, email="Bob@Example.com")
    response = register(client, email="bob@example.com ")
    assert response.status_code == 400 and b"already exists" in response.data


@pytest.mark.parametrize("data,message", [
    ({"name": "A", "email": "not-an-email", "password": PASSWORD, "confirm": PASSWORD}, b"valid email"),
    ({"name": "A", "email": "a@b.co", "password": "short", "confirm": "short"}, b"8-128"),
    ({"name": "A", "email": "a@b.co", "password": PASSWORD, "confirm": "different-1"}, b"do not match"),
    ({"name": "", "email": "a@b.co", "password": PASSWORD, "confirm": PASSWORD}, b"required"),
])
def test_registration_validation(client, data, message):
    response = client.post("/register", data=data)
    assert response.status_code == 400 and message in response.data


def test_wrong_password_and_unknown_user_look_identical(client):
    register(client)
    wrong = login(client, password="wrong-password")
    unknown = login(client, email="nobody@example.com")
    assert wrong.status_code == unknown.status_code == 401
    assert b"Incorrect email or password" in wrong.data and b"Incorrect email or password" in unknown.data


def test_logout_requires_post(alice):
    assert alice.get("/logout").status_code == 405
    assert alice.post("/logout").status_code == 302
    assert alice.get("/upload").status_code == 302  # back to login


def test_open_redirect_blocked(client):
    register(client)
    response = client.post("/login?next=https://evil.example/", data={"email": "alice@example.com", "password": PASSWORD})
    assert response.headers["Location"] == "/upload"
    response = client.post("/logout")
    response = client.post("/login?next=/history", data={"email": "alice@example.com", "password": PASSWORD})
    assert response.headers["Location"] == "/history"

# ---------- unauthenticated access ----------

@pytest.mark.parametrize("path", ["/upload", "/history", "/scan/1", "/scan/1/image/input", "/report/1", "/admin/"])
def test_private_pages_require_login(client, path):
    response = client.get(path)
    assert response.status_code == 302 and "/login" in response.headers["Location"]


def test_predict_requires_login(client, png):
    assert upload(client, png, accept_json=False).status_code == 302

# ---------- prediction ----------

def test_successful_prediction_creates_all_records(alice, app, png):
    response = upload(alice, png, filename="../../etc/passwd.png")
    assert response.status_code == 201
    body = response.get_json()
    assert body["label"] in ("Normal", "Osteopenia", "Osteoporosis")
    assert sum(body["scores"].values()) == pytest.approx(1.0, abs=1e-4)
    assert body["model_version"] == "tiny-test-model"
    with app.app_context():
        scan = db.session.get(Scan, body["scan_id"])
        assert scan.status == "completed"
        assert ".." not in scan.image_path and re.fullmatch(r"originals/[0-9a-f]{32}\.png", scan.image_path)
        assert scan.original_filename == "../../etc/passwd.png"  # kept as text only
        assert scan.prediction.report.pdf_path.endswith(".pdf")
        assert scan.prediction.confidence == max(scan.prediction.scores.values())
        assert (app.config["UPLOADS_DIR"] / scan.image_path).is_file()


def test_result_page_and_images(alice, png):
    scan_id = upload(alice, png).get_json()["scan_id"]
    page = alice.get(f"/scan/{scan_id}")
    assert page.status_code == 200 and b"not a diagnosis" in page.data and b"tiny-test-model" in page.data
    image = alice.get(f"/scan/{scan_id}/image/input")
    assert image.status_code == 200 and image.data.startswith(b"\x89PNG")
    assert image.headers["Cache-Control"] == "no-store"
    assert alice.get(f"/scan/{scan_id}/image/secret").status_code == 404


def test_overlay_controls(alice, png):
    scan_id = upload(alice, png).get_json()["scan_id"]
    ok = alice.get(f"/scan/{scan_id}/overlay?alpha=0.7&colormap=viridis")
    assert ok.status_code == 200 and ok.data.startswith(b"\x89PNG")
    assert alice.get(f"/scan/{scan_id}/overlay?alpha=7").status_code == 400
    assert alice.get(f"/scan/{scan_id}/overlay?alpha=abc").status_code == 400
    assert alice.get(f"/scan/{scan_id}/overlay?colormap=../../x").status_code == 400


def test_html_form_upload_redirects_to_result(alice, png):
    response = upload(alice, png, accept_json=False)
    assert response.status_code == 302 and "/scan/" in response.headers["Location"]


@pytest.mark.parametrize("data,message", [
    (b"", "empty"),
    (b"just some text pretending to be an image", "Unsupported"),
    (b"%PDF-1.7\n fake pdf", "Unsupported"),
])
def test_invalid_uploads_rejected(alice, app, data, message):
    response = upload(alice, data)
    assert response.status_code == 400 and message in response.get_json()["error"]
    with app.app_context():
        assert Scan.query.count() == 0  # nothing stored for rejected files


def test_corrupt_image_rejected(alice, png):
    response = upload(alice, png[:60])
    assert response.status_code == 400 and "corrupt" in response.get_json()["error"]


def test_oversized_upload_rejected(alice):
    response = upload(alice, b"\x89PNG\r\n\x1a\n" + b"0" * (3 * 1024 * 1024))
    assert response.status_code == 413


def test_16bit_upload_accepted(alice):
    image16 = (make_xray_like(9).astype(np.uint16) * 16)
    assert upload(alice, encode(image16)).status_code == 201


def test_missing_file_field(alice):
    response = alice.post("/predict", data={}, headers={"Accept": "application/json"})
    assert response.status_code == 400


def test_inference_failure_marks_scan_failed_without_leaking_details(alice, app, png, monkeypatch):
    from app.services.inference import inference_service

    def explode(*args, **kwargs):
        raise RuntimeError("secret internal detail /path/to/model")

    monkeypatch.setattr(inference_service, "predict", explode)
    response = upload(alice, png)
    assert response.status_code == 500
    assert "secret" not in response.get_data(as_text=True)
    with app.app_context():
        scan = Scan.query.one()
        assert scan.status == "failed" and "secret" not in scan.error_message
        assert Prediction.query.count() == 0

# ---------- history and ownership ----------

def test_history_lists_only_own_scans(app, png):
    alice, bob = app.test_client(), app.test_client()
    register(alice); login(alice)
    register(bob, email="bob@example.com", name="Bob"); login(bob, email="bob@example.com")
    upload(alice, png)
    upload(alice, png)
    upload(bob, png)
    alice_history = alice.get("/history").get_data(as_text=True)
    bob_history = bob.get("/history").get_data(as_text=True)
    assert alice_history.count("S-0000") == 2 and bob_history.count("S-0000") == 1


def test_cross_user_access_is_blocked_everywhere(app, png):
    alice, bob = app.test_client(), app.test_client()
    register(alice); login(alice)
    register(bob, email="bob@example.com", name="Bob"); login(bob, email="bob@example.com")
    scan_id = upload(alice, png).get_json()["scan_id"]
    with app.app_context():
        report_id = db.session.get(Scan, scan_id).prediction.report.id
    for path in [f"/scan/{scan_id}", f"/scan/{scan_id}/image/input", f"/scan/{scan_id}/image/overlay",
                 f"/scan/{scan_id}/overlay", f"/report/{report_id}"]:
        assert bob.get(path).status_code == 404, path  # 404, not 403: existence not revealed
        assert alice.get(path).status_code == 200, path

# ---------- PDF reports ----------

def test_pdf_download(alice, png):
    scan_id = upload(alice, png).get_json()["scan_id"]
    page = alice.get(f"/scan/{scan_id}").get_data(as_text=True)
    report_url = re.search(r'href="(/report/\d+)"', page).group(1)
    response = alice.get(report_url)
    assert response.status_code == 200 and response.mimetype == "application/pdf"
    assert response.data.startswith(b"%PDF")
    assert "attachment" in response.headers["Content-Disposition"]


def test_missing_pdf_is_regenerated(alice, app, png):
    scan_id = upload(alice, png).get_json()["scan_id"]
    with app.app_context():
        report = db.session.get(Scan, scan_id).prediction.report
        (app.config["REPORTS_DIR"] / report.pdf_path).unlink()
        report_id = report.id
    assert alice.get(f"/report/{report_id}").data.startswith(b"%PDF")


def test_pdf_builder_handles_missing_images(tmp_path):
    from datetime import datetime

    from app.services.reports import build_pdf

    target = tmp_path / "r.pdf"
    build_pdf(target, report_id=1, scan_id=2, predicted_at=datetime(2026, 1, 1), label="Normal",
              scores={"Normal": 0.5, "Osteopenia": 0.3, "Osteoporosis": 0.2}, model_version="v",
              preprocessing_version="p", original_image=tmp_path / "missing.png", overlay_image=None,
              heatmap_informative=False)
    assert target.read_bytes().startswith(b"%PDF")

# ---------- database relationships ----------

def test_relationships_and_cascade(app):
    with app.app_context():
        user = User(name="C", email="c@example.com")
        user.set_password(PASSWORD)
        scan = Scan(user=user, image_path="originals/x.png", status="completed")
        prediction = Prediction(scan=scan, label="Normal", confidence=0.6, score_normal=0.6,
                                score_osteopenia=0.3, score_osteoporosis=0.1, model_version="v",
                                preprocessing_version="p")
        db.session.add_all([user, scan, prediction, Report(prediction=prediction)])
        db.session.commit()
        assert user.scans[0].prediction.report.prediction is prediction
        db.session.delete(user)
        db.session.commit()
        assert Scan.query.count() == Prediction.query.count() == Report.query.count() == 0


def test_invalid_status_rejected_by_database(app):
    from sqlalchemy.exc import IntegrityError

    with app.app_context():
        user = User(name="D", email="d@example.com", password_hash="x")
        db.session.add_all([user, Scan(user=user, image_path="a", status="weird")])
        with pytest.raises(IntegrityError):
            db.session.commit()

# ---------- admin ----------

def test_admin_requires_admin_role(app, alice):
    assert alice.get("/admin/").status_code == 403
    with app.app_context():
        User.query.filter_by(email="alice@example.com").one().role = "admin"
        db.session.commit()
    page = alice.get("/admin/")
    assert page.status_code == 200 and b"tiny-test-model" in page.data


def test_create_admin_cli(app):
    result = app.test_cli_runner().invoke(args=["create-admin", "--email", "Admin@Example.com", "--name", "Root",
                                                "--password", PASSWORD])
    assert "Administrator ready" in result.output
    with app.app_context():
        assert User.query.filter_by(email="admin@example.com").one().is_admin

# ---------- missing model ----------

def test_missing_model_fails_gracefully(tmp_path, png):
    app = create_app("testing", {"UPLOADS_DIR": tmp_path / "u", "REPORTS_DIR": tmp_path / "r",
                                 "LOG_DIR": tmp_path / "l", "MODEL_DIR": tmp_path / "no-model"})
    client = app.test_client()
    assert client.get("/health").get_json()["model_loaded"] is False
    register(client); login(client)
    page = client.get("/upload")
    assert b"Analysis is unavailable" in page.data
    response = upload(client, png)
    assert response.status_code == 503 and "not installed" in response.get_json()["error"]

# ---------- security ----------

def test_csrf_enforced_when_enabled(tmp_path, tiny_model_dir):
    app = create_app("testing", {"WTF_CSRF_ENABLED": True, "UPLOADS_DIR": tmp_path / "u",
                                 "REPORTS_DIR": tmp_path / "r", "LOG_DIR": tmp_path / "l",
                                 "MODEL_DIR": tiny_model_dir, "LOAD_MODEL_ON_STARTUP": False})
    client = app.test_client()
    assert register(client).status_code == 400  # no token
    page = client.get("/register").get_data(as_text=True)
    token = re.search(r'name="csrf_token" type="hidden" value="([^"]+)"', page).group(1)
    response = client.post("/register", data={"csrf_token": token, "name": "E", "email": "e@example.com",
                                              "password": PASSWORD, "confirm": PASSWORD})
    assert response.status_code == 302


def test_security_headers(client):
    headers = client.get("/login").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert "default-src 'self'" in headers["Content-Security-Policy"]


def test_production_requires_secret_key(monkeypatch):
    from app.settings import ProductionSettings

    monkeypatch.setattr(ProductionSettings, "SECRET_KEY", None)
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        create_app("production", {"LOAD_MODEL_ON_STARTUP": False})


def test_storage_rejects_path_escape(tmp_path):
    from app.services.storage import StoragePathError, resolve_inside

    with pytest.raises(StoragePathError):
        resolve_inside(tmp_path, "../outside.txt")
    assert resolve_inside(tmp_path, "a/b.png") == (tmp_path / "a" / "b.png").resolve()


def test_unknown_page_returns_friendly_404(client):
    response = client.get("/definitely-not-here")
    assert response.status_code == 404 and b"does not exist" in response.data and b"Traceback" not in response.data


def test_retention_purge(alice, app, png):
    from datetime import timedelta

    from app.services.scans import purge_older_than

    scan_id = upload(alice, png).get_json()["scan_id"]
    with app.app_context():
        scan = db.session.get(Scan, scan_id)
        original = app.config["UPLOADS_DIR"] / scan.image_path
        scan.uploaded_at = scan.uploaded_at - timedelta(days=100)
        db.session.commit()
        assert purge_older_than(90) == 1
        assert not original.exists() and Scan.query.count() == 0


# ---------- hybrid mode (X-ray + patient details) ----------

PATIENT = {"use_clinical": "y", "age": "68", "sex": "female", "height_cm": "152", "weight_kg": "55",
           "menopause_age": "47", "smoker": "y", "family_history": "y"}


def upload_with(client, data: bytes, fields: dict):
    import io

    return client.post("/predict", data={"image": (io.BytesIO(data), "knee.png"), **fields},
                       content_type="multipart/form-data", headers={"Accept": "application/json"})


def test_hybrid_prediction_stores_inputs_and_components(alice, app, png):
    response = upload_with(alice, png, PATIENT)
    assert response.status_code == 201
    body = response.get_json()
    assert body["mode"] == "hybrid" and sum(body["scores"].values()) == pytest.approx(1.0, abs=1e-4)
    with app.app_context():
        p = db.session.get(Scan, body["scan_id"]).prediction
        assert p.mode == "hybrid"
        assert p.clinical_inputs_dict["age"] == 68 and p.clinical_inputs_dict["smoker"] is True
        assert set(p.component_scores_dict) == {"image", "clinical", "osteoporosis_boost"}
    page = alice.get(f"/scan/{body['scan_id']}").get_data(as_text=True)
    assert "X-ray + patient details" in page and "How the combined score was formed" in page
    report_url = re.search(r'href="(/report/\d+)"', page).group(1)
    assert alice.get(report_url).data.startswith(b"%PDF")


def test_image_only_mode_warns_low_reliability(alice, png):
    body = upload(alice, png).get_json()
    assert body["mode"] == "image_only"
    page = alice.get(f"/scan/{body['scan_id']}").get_data(as_text=True)
    assert "Low reliability" in page and "X-ray only" in page


def test_unticked_clinical_box_ignores_fields(alice, png):
    fields = dict(PATIENT)
    fields.pop("use_clinical")
    assert upload_with(alice, png, fields).get_json()["mode"] == "image_only"


@pytest.mark.parametrize("change,message", [
    ({"age": ""}, "Required when patient details"),
    ({"height_cm": "500"}, "between 100"),
    ({"sex": "male"}, "Only applies to female"),
    ({"menopause_age": "70", "age": "60"}, "between 25"),
    ({"menopause_age": "55", "age": "50"}, "later than the current age"),
    ({"sex": "other"}, "Not a valid choice"),
])
def test_clinical_validation(alice, app, png, change, message):
    response = upload_with(alice, png, {**PATIENT, **change})
    assert response.status_code == 400 and message in response.get_json()["error"]
    with app.app_context():
        assert Scan.query.count() == 0


def test_hybrid_rejected_when_model_has_no_clinical_part(tmp_path, image_only_model_dir, png):
    app = create_app("testing", {"UPLOADS_DIR": tmp_path / "u", "REPORTS_DIR": tmp_path / "r",
                                 "LOG_DIR": tmp_path / "l", "MODEL_DIR": image_only_model_dir})
    client = app.test_client()
    register(client); login(client)
    assert client.get("/health").get_json()["hybrid_available"] is False
    assert b"Add patient details" not in client.get("/upload").data
    response = upload_with(client, png, PATIENT)
    assert response.status_code == 400 and "no clinical component" in response.get_json()["error"]
    assert upload(client, png).status_code == 201  # image-only still works


def test_schema_upgrade_adds_columns_to_old_database(tmp_path, tiny_model_dir):
    import sqlite3

    db_file = tmp_path / "old.db"
    con = sqlite3.connect(db_file)  # the predictions table as it was before hybrid mode
    con.executescript("""
        CREATE TABLE predictions (id INTEGER PRIMARY KEY, scan_id INTEGER NOT NULL UNIQUE, label VARCHAR(50) NOT NULL,
            confidence FLOAT NOT NULL, score_normal FLOAT NOT NULL, score_osteopenia FLOAT NOT NULL,
            score_osteoporosis FLOAT NOT NULL, model_version VARCHAR(100) NOT NULL,
            preprocessing_version VARCHAR(50) NOT NULL, heatmap_informative BOOLEAN NOT NULL, predicted_at DATETIME NOT NULL);
        INSERT INTO predictions VALUES (1, 1, 'Normal', 0.5, 0.5, 0.3, 0.2, 'v', 'p', 0, '2026-01-01 00:00:00');
    """)
    con.commit(); con.close()
    create_app("testing", {"SQLALCHEMY_DATABASE_URI": f"sqlite:///{db_file.as_posix()}", "UPLOADS_DIR": tmp_path / "u",
                           "REPORTS_DIR": tmp_path / "r", "LOG_DIR": tmp_path / "l", "LOAD_MODEL_ON_STARTUP": False})
    con = sqlite3.connect(db_file)
    columns = {row[1] for row in con.execute("PRAGMA table_info(predictions)")}
    assert {"mode", "clinical_inputs", "component_scores"} <= columns
    assert con.execute("SELECT mode, label FROM predictions").fetchone() == ("image_only", "Normal")  # row kept
    con.close()


def test_patient_details_ticked_by_default(alice):
    page = alice.get("/upload").get_data(as_text=True)
    checkbox = re.search(r'<input[^>]*id="use_clinical"[^>]*>', page).group(0)
    assert "checked" in checkbox
    fieldset = re.search(r'<fieldset id="clinical-fields"[^>]*>', page).group(0)
    assert "hidden" not in fieldset


def test_ticked_box_with_empty_details_asks_for_them(alice, png):
    response = upload_with(alice, png, {"use_clinical": "y"})
    assert response.status_code == 400 and "Required when patient details" in response.get_json()["error"]
