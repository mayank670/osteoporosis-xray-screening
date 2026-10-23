"""Registration, login and logout."""

from __future__ import annotations

import logging
from urllib.parse import urlsplit

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user
from sqlalchemy.exc import IntegrityError

from app.extensions import db
from app.forms import LoginForm, LogoutForm, RegistrationForm
from app.models import User

bp = Blueprint("auth", __name__)
log = logging.getLogger(__name__)


def _safe_next(target: str | None) -> str:
    """Only allow redirects to paths on this site (prevents 'open redirect' attacks)."""
    if target:
        parts = urlsplit(target)
        if not parts.scheme and not parts.netloc and target.startswith("/") and not target.startswith("//"):
            return target
    return url_for("main.upload")


@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.upload"))
    form = RegistrationForm()
    if form.validate_on_submit():
        email = form.email.data.strip().lower()
        if User.query.filter_by(email=email).first():
            form.email.errors.append("An account with this email already exists.")
        else:
            user = User(name=form.name.data.strip(), email=email, role="user")
            user.set_password(form.password.data)
            db.session.add(user)
            try:
                db.session.commit()
            except IntegrityError:  # two simultaneous registrations with the same email
                db.session.rollback()
                form.email.errors.append("An account with this email already exists.")
            else:
                log.info("New user registered: id=%s", user.id)
                flash("Account created. Please log in.", "success")
                return redirect(url_for("auth.login"))
    return render_template("auth/register.html", form=form), 400 if request.method == "POST" else 200


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.upload"))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data.strip().lower()).first()
        # Same message whether the email or the password is wrong: does not reveal which accounts exist.
        if user is None or not user.check_password(form.password.data):
            log.warning("Failed login attempt")
            flash("Incorrect email or password.", "error")
            return render_template("auth/login.html", form=form), 401
        login_user(user)
        return redirect(_safe_next(request.args.get("next")))
    return render_template("auth/login.html", form=form)


@bp.post("/logout")
@login_required
def logout():
    if LogoutForm().validate_on_submit():
        logout_user()
        flash("You have been logged out.", "success")
    return redirect(url_for("auth.login"))
