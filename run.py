"""Development server:  python run.py   (http://127.0.0.1:5000)

Not for production — use `python serve.py` (Waitress) instead.
"""

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=app.config["DEBUG"], use_reloader=False)
