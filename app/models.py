"""Database tables (SQLAlchemy ORM). Images and PDFs live on disk; rows store paths.

User 1 --- * Scan 1 --- 0..1 Prediction 1 --- 0..1 Report
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from app.extensions import db


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)  # stored as naive UTC


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(255), nullable=False, unique=True, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="user")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    scans = db.relationship("Scan", back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (db.CheckConstraint("role IN ('user', 'admin')", name="ck_users_role"),)

    def set_password(self, password: str) -> None:
        # Werkzeug uses a salted, slow hash (scrypt); the plain password is never stored.
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


class Scan(db.Model):
    __tablename__ = "scans"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    image_path = db.Column(db.String(255), nullable=False)  # relative to UPLOADS_DIR, server-generated
    display_path = db.Column(db.String(255))  # PNG preview in model geometry
    original_filename = db.Column(db.String(255))  # shown to the user only, never used as a path
    uploaded_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    status = db.Column(db.String(20), nullable=False, default="processing")
    error_message = db.Column(db.String(500))  # user-safe text, no stack traces

    user = db.relationship("User", back_populates="scans")
    prediction = db.relationship("Prediction", back_populates="scan", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        db.CheckConstraint("status IN ('processing', 'completed', 'failed')", name="ck_scans_status"),
    )


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    scan_id = db.Column(db.Integer, db.ForeignKey("scans.id", ondelete="CASCADE"), nullable=False, unique=True)
    label = db.Column(db.String(50), nullable=False)
    # Highest softmax output. A model score, NOT a calibrated probability of disease.
    confidence = db.Column(db.Float, nullable=False)
    score_normal = db.Column(db.Float, nullable=False)
    score_osteopenia = db.Column(db.Float, nullable=False)
    score_osteoporosis = db.Column(db.Float, nullable=False)
    model_version = db.Column(db.String(100), nullable=False, index=True)
    preprocessing_version = db.Column(db.String(50), nullable=False)
    heatmap_informative = db.Column(db.Boolean, nullable=False, default=False)
    # "image_only" (X-ray CNN alone) or "hybrid" (X-ray + patient details)
    mode = db.Column(db.String(20), nullable=False, default="image_only")
    clinical_inputs = db.Column(db.Text)  # JSON: the patient details used (hybrid only)
    component_scores = db.Column(db.Text)  # JSON: {"image": {...}, "clinical": {...}}
    predicted_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    scan = db.relationship("Scan", back_populates="prediction")
    report = db.relationship("Report", back_populates="prediction", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        db.CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_predictions_conf"),
        db.CheckConstraint("mode IN ('image_only', 'hybrid')", name="ck_predictions_mode"),
    )

    @property
    def scores(self) -> dict[str, float]:
        return {"Normal": self.score_normal, "Osteopenia": self.score_osteopenia,
                "Osteoporosis": self.score_osteoporosis}

    @property
    def clinical_inputs_dict(self) -> dict:
        return json.loads(self.clinical_inputs) if self.clinical_inputs else {}

    @property
    def component_scores_dict(self) -> dict:
        return json.loads(self.component_scores) if self.component_scores else {}


class Report(db.Model):
    __tablename__ = "reports"

    id = db.Column(db.Integer, primary_key=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey("predictions.id", ondelete="CASCADE"),
                              nullable=False, unique=True)
    heatmap_path = db.Column(db.String(255))  # overlay PNG, relative to UPLOADS_DIR
    pdf_path = db.Column(db.String(255))  # relative to REPORTS_DIR
    generated_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    prediction = db.relationship("Prediction", back_populates="report")
