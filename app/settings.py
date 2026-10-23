"""Flask settings per environment. Select with APP_ENV=development|production."""

from __future__ import annotations

import os
import secrets
from datetime import timedelta

import config


class BaseSettings:
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{(config.PROJECT_ROOT / 'instance' / 'app.db').as_posix()}")
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024  # Flask returns 413 above this
    SESSION_COOKIE_HTTPONLY = True  # JavaScript cannot read the session cookie
    SESSION_COOKIE_SAMESITE = "Lax"  # not sent on cross-site POSTs (CSRF defence in depth)
    PERMANENT_SESSION_LIFETIME = timedelta(hours=2)
    REMEMBER_COOKIE_HTTPONLY = True
    MODEL_DIR = config.MODEL_DIR
    UPLOADS_DIR = config.UPLOADS_DIR
    REPORTS_DIR = config.REPORTS_DIR
    LOG_DIR = config.PROJECT_ROOT / "instance" / "logs"
    LOAD_MODEL_ON_STARTUP = True
    RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "90"))


class DevelopmentSettings(BaseSettings):
    DEBUG = True
    # A random key is fine for local development: sessions just end on restart.
    SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)


class ProductionSettings(BaseSettings):
    DEBUG = False
    SESSION_COOKIE_SECURE = True  # cookie only over HTTPS
    REMEMBER_COOKIE_SECURE = True
    SECRET_KEY = os.environ.get("SECRET_KEY")  # checked in create_app: must be set


class TestingSettings(BaseSettings):
    TESTING = True
    SECRET_KEY = "test-only-not-secret"
    SQLALCHEMY_DATABASE_URI = "sqlite://"  # in-memory database
    WTF_CSRF_ENABLED = False  # re-enabled in the dedicated CSRF test


SETTINGS = {"development": DevelopmentSettings, "production": ProductionSettings, "testing": TestingSettings}
