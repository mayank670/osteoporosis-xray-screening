"""Command-line tasks:  flask --app run <command>

    create-admin   create (or promote) an administrator account
    purge-scans    delete scans and files older than RETENTION_DAYS
"""

from __future__ import annotations

import click
from flask import Flask


def register_cli(app: Flask) -> None:
    @app.cli.command("create-admin")
    @click.option("--email", prompt=True)
    @click.option("--name", prompt=True, default="Administrator")
    @click.password_option()
    def create_admin(email: str, name: str, password: str) -> None:
        from app.extensions import db
        from app.models import User

        if len(password) < 8:
            raise click.BadParameter("Password must be at least 8 characters.")
        email = email.strip().lower()
        user = User.query.filter_by(email=email).first()
        if user is None:
            user = User(name=name, email=email)
            db.session.add(user)
        user.role = "admin"
        user.set_password(password)
        db.session.commit()
        click.echo(f"Administrator ready: {email}")

    @app.cli.command("purge-scans")
    @click.option("--days", type=int, default=None, help="default: RETENTION_DAYS setting")
    def purge_scans(days: int | None) -> None:
        from app.services.scans import purge_older_than

        days = days if days is not None else app.config["RETENTION_DAYS"]
        click.echo(f"Deleted {purge_older_than(days)} scans older than {days} days.")
