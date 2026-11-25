"""
WEB PAGES ROUTES
----------------
Serves the unified inspection frontend template with route support for:
- / and /overview (Application Overview)
- /inspection/bearing (Bearing Inspection Workspace)
- /inspection/ring (Ring Inspection Workspace)
- /inspection/plastic (Plastic Defect Inspection Workspace)
Conforms to ui_ux_requirements.md Part 2.3.
"""
from flask import Blueprint, render_template

pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
@pages_bp.route("/landing")
def landing_page():
    return render_template("landing.html")


@pages_bp.route("/overview")
def overview():
    return render_template("index.html", initial_view="overview")


@pages_bp.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")


@pages_bp.route("/inspection")
@pages_bp.route("/inspection/ring")
def ring_workspace():
    return render_template("index.html", initial_view="ring")


@pages_bp.route("/inspection/bearing")
def bearing_workspace():
    return render_template("index.html", initial_view="bearing")


@pages_bp.route("/inspection/plastic")
def plastic_workspace():
    return render_template("index.html", initial_view="plastic")
