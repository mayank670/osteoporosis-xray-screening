"""Application factory: create_app() builds and configures a Flask app instance.

A factory (instead of a global `app = Flask(...)`) lets tests create isolated
apps with their own database and model directory.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from flask import Flask, render_template, request

from app.extensions import csrf, db, login_manager
from app.settings import SETTINGS


def create_app(env: str | None = None, overrides: dict | None = None) -> Flask:
    env = env or os.environ.get("APP_ENV", "development")
    app = Flask(__name__, instance_path=str(Path(__file__).resolve().parents[1] / "instance"))
    app.config.from_object(SETTINGS[env])
    app.config.update(overrides or {})
    if not app.config.get("SECRET_KEY"):
        raise RuntimeError("SECRET_KEY must be set (see .env.example).")

    for key in ("UPLOADS_DIR", "REPORTS_DIR", "LOG_DIR"):
        Path(app.config[key]).mkdir(parents=True, exist_ok=True)
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    _configure_logging(app)

    db.init_app(app)
    csrf.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please log in to continue."
    login_manager.session_protection = "strong"  # session dropped if IP/user-agent fingerprint changes

    from app.models import User

    @login_manager.user_loader
    def load_user(user_id: str):
        return db.session.get(User, int(user_id))

    from app.routes.admin import bp as admin_bp
    from app.routes.auth import bp as auth_bp
    from app.routes.main import bp as main_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(admin_bp)

    from app.cli import register_cli

    register_cli(app)
    _register_error_handlers(app)
    _register_security_headers(app)

    with app.app_context():
        db.create_all()  # creates missing tables; see docs/architecture.md for schema changes
        _upgrade_schema()

    if app.config["LOAD_MODEL_ON_STARTUP"]:
        from app.services.inference import inference_service

        inference_service.load(app.config["MODEL_DIR"])

    @app.context_processor
    def inject_globals():
        from app.forms import LogoutForm
        from app.services.inference import inference_service

        return {"logout_form": LogoutForm(), "model_ready": inference_service.ready,
                "model_version": inference_service.model_version,
                "heldout": inference_service.metadata.get("heldout_performance", {}),
                "heldout_hybrid": (inference_service.metadata.get("heldout_performance", {}).get("hybrid_screening")
                                   or inference_service.metadata.get("heldout_performance", {}).get("hybrid")),
                "fusion_image_weight": (inference_service.predictor.clinical_model.fusion_image_weight
                                        if inference_service.ready and inference_service.predictor.supports_clinical
                                        else None)}

    return app


# Columns added after the first release. create_all() never alters an existing
# table, so a database created earlier gets them here (SQLite ALTER TABLE ADD
# COLUMN keeps all existing rows). A full project would use Alembic instead.
_ADDED_COLUMNS = {
    "predictions": {
        "mode": "VARCHAR(20) NOT NULL DEFAULT 'image_only'",
        "clinical_inputs": "TEXT",
        "component_scores": "TEXT",
    },
}


def _upgrade_schema() -> None:
    from sqlalchemy import inspect, text

    inspector = inspect(db.engine)
    for table, columns in _ADDED_COLUMNS.items():
        if not inspector.has_table(table):
            continue
        existing = {c["name"] for c in inspector.get_columns(table)}
        for name, ddl in columns.items():
            if name not in existing:
                with db.engine.begin() as connection:
                    connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
                logging.getLogger(__name__).info("Schema upgrade: added %s.%s", table, name)


def _configure_logging(app: Flask) -> None:
    handler = RotatingFileHandler(Path(app.config["LOG_DIR"]) / "app.log", maxBytes=1_000_000,
                                  backupCount=5, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        root.addHandler(handler)
    root.setLevel(logging.INFO)


def _register_error_handlers(app: Flask) -> None:
    def wants_json() -> bool:
        return request.accept_mimetypes.best == "application/json" or request.path == "/health"

    def render_error(status: int, title: str, message: str):
        if wants_json():
            return {"error": message}, status
        return render_template("error.html", status=status, title=title, message=message), status

    @app.errorhandler(400)
    def bad_request(_):
        return render_error(400, "Bad request", "The request could not be processed. Please try again.")

    @app.errorhandler(403)
    def forbidden(_):
        return render_error(403, "Forbidden", "You do not have permission to view this page.")

    @app.errorhandler(404)
    def not_found(_):
        return render_error(404, "Not found", "The page or file you asked for does not exist.")

    @app.errorhandler(413)
    def too_large(_):
        limit = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return render_error(413, "File too large", f"The file is larger than the {limit} MB limit.")

    @app.errorhandler(500)
    def server_error(_):
        # The traceback goes to the log; the browser gets a generic message only.
        return render_error(500, "Server error", "Something went wrong on our side. The error has been logged.")


def _register_security_headers(app: Flask) -> None:
    @app.after_request
    def add_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; "
            "frame-ancestors 'none'")
        if request.endpoint != "static":
            # Overwrite (send_file sets "no-cache", which still allows storing private X-rays).
            response.headers["Cache-Control"] = "no-store"
        return response
