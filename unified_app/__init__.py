"""
UNIFIED INSPECTION FLASK APPLICATION FACTORY
--------------------------------------------
"""
from pathlib import Path
from flask import Flask
from .routes import pages_bp, api_bp
from .config import MAX_CONTENT_LENGTH


def create_app():
    app_root = Path(__file__).resolve().parent

    app = Flask(
        __name__,
        template_folder=str(app_root / "templates"),
        static_folder=str(app_root / "static")
    )

    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    # Register blueprints
    app.register_blueprint(pages_bp)
    app.register_blueprint(api_bp)

    return app
