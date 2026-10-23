"""Extension objects are created here, unbound, and attached in create_app().

This avoids circular imports: models and routes import `db` from here, not
from app/__init__.py.
"""

from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import CSRFProtect

db = SQLAlchemy()
login_manager = LoginManager()
csrf = CSRFProtect()
