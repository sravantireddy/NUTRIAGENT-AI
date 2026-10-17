"""
NutriAgent AI — Main Flask application entry point.
IBM Granite-powered multi-agent nutrition system.
"""
from __future__ import annotations
import logging
import os
from pathlib import Path

from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

from config import FLASK_SECRET_KEY, FLASK_DEBUG, PORT
from routes import api_bp

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("nutriagent")


# ── Application factory ────────────────────────────────────────────────────────
def create_app() -> Flask:
    static_folder = Path(__file__).parent / "static"
    app = Flask(__name__, static_folder=str(static_folder), static_url_path="/static")
    app.secret_key = FLASK_SECRET_KEY

    # CORS — allow all origins in dev; restrict in production
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register API blueprint
    app.register_blueprint(api_bp)

    # ── Serve the single-page dashboard ───────────────────────────────────────
    @app.route("/")
    def index():
        return send_from_directory(str(static_folder), "index.html")

    @app.route("/<path:path>")
    def static_files(path):
        try:
            return send_from_directory(str(static_folder), path)
        except Exception:
            return send_from_directory(str(static_folder), "index.html")

    # ── Global error handlers ──────────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found", "status": 404}), 404

    @app.errorhandler(500)
    def internal_error(e):
        logger.error("Internal server error: %s", e)
        return jsonify({"error": "Internal server error", "status": 500}), 500

    logger.info("NutriAgent AI application created successfully.")
    logger.info("   Agents: Nutrition Knowledge | Diet Recommendation | Health Advisory | Food Log & Feedback")
    logger.info("   Model:  %s (IBM watsonx.ai)", "meta-llama/llama-3-3-70b-instruct")
    return app


# ── Entry point ────────────────────────────────────────────────────────────────
app = create_app()

if __name__ == "__main__":
    logger.info("🚀 Starting NutriAgent AI on http://localhost:%d", PORT)
    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=FLASK_DEBUG,
        use_reloader=False,   # Disable reloader to avoid double-init of agents
        threaded=True,        # Handle concurrent requests (each in its own thread)
    )
