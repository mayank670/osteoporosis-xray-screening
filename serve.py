"""Production server using Waitress (a pure-Python WSGI server that runs on Windows).

    set APP_ENV=production
    set SECRET_KEY=<long random value>
    python serve.py

Put it behind a reverse proxy that terminates HTTPS (session cookies are
marked Secure in production, so plain HTTP logins will not work).
"""

import os

from waitress import serve

from app import create_app

if __name__ == "__main__":
    os.environ.setdefault("APP_ENV", "production")
    app = create_app()
    # Few threads: the model is CPU-heavy and predictions are serialised by a lock anyway.
    serve(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")), threads=4)
