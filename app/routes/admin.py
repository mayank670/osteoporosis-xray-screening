"""Administrator dashboard: usage statistics, recent failures, model information.

Admins see aggregate numbers and error messages, NOT other users' images or
reports — privacy by design.
"""

from __future__ import annotations

from functools import wraps

from flask import Blueprint, abort, render_template
from flask_login import current_user, login_required
from sqlalchemy import func

from app.extensions import db
from app.models import Prediction, Scan, User
from app.services.inference import inference_service

bp = Blueprint("admin", __name__, url_prefix="/admin")


def admin_required(view):
    @wraps(view)  # keeps the original function name, which Flask uses for the endpoint
    @login_required
    def wrapper(*args, **kwargs):
        if not current_user.is_admin:
            abort(403)
        return view(*args, **kwargs)
    return wrapper


@bp.get("/")
@admin_required
def dashboard():
    status_counts = dict(db.session.query(Scan.status, func.count(Scan.id)).group_by(Scan.status).all())
    label_counts = dict(db.session.query(Prediction.label, func.count(Prediction.id))
                        .group_by(Prediction.label).all())
    version_counts = (db.session.query(Prediction.model_version, func.count(Prediction.id))
                      .group_by(Prediction.model_version).all())
    failures = (Scan.query.filter_by(status="failed").order_by(Scan.uploaded_at.desc()).limit(20).all())
    return render_template(
        "admin.html",
        user_count=User.query.count(),
        status_counts=status_counts,
        label_counts=label_counts,
        version_counts=version_counts,
        failures=failures,
        model_ready=inference_service.ready,
        model_error=inference_service.error,
        metadata=inference_service.metadata,
    )
