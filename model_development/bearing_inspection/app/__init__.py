"""
Flask Application Factory for Casting Defect Detector.
"""
from flask import Flask, jsonify
import config
from app.storage import init_db


def create_app() -> Flask:
    """Creates and configures the Flask application."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
    app.config["SECRET_KEY"] = "casting-inspection-secret-key-2026"

    # Initialize storage tables
    try:
        init_db()
    except Exception as e:
        print(f"Warning: Failed to initialize SQLite storage: {e}")

    # Register blueprints
    from app.routes import bp
    app.register_blueprint(bp)

    @app.errorhandler(413)
    def request_entity_too_large(error):
        max_mb = int(config.MAX_CONTENT_LENGTH / (1024 * 1024))
        return jsonify({
            "status": "invalid_input",
            "module": "casting_defect",
            "error_message": f"Ukuran file melebihi batas maksimum {max_mb} MB.",
            "warnings": [f"Batas upload adalah {max_mb} MB."]
        }), 413

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "status": "error",
            "error_message": "Endpoint tidak ditemukan."
        }), 404

    @app.errorhandler(500)
    def internal_server_error(error):
        return jsonify({
            "status": "error",
            "error_message": "Terjadi kesalahan internal pada server."
        }), 500

    return app

