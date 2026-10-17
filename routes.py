"""
Flask REST API — routes for all four nutrition agents.
Mounted at /api/v1/*
"""
from __future__ import annotations
import logging
from flask import Blueprint, request, jsonify

from agents import (
    NutritionKnowledgeAgent,
    DietRecommendationAgent,
    HealthAdvisoryAgent,
    FoodLogFeedbackAgent,
)
from agents.diet_recommendation_agent import UserProfile
from rag_store import build_rag_store

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api/v1")

# ── Singleton agent instances (created once at blueprint load) ─────────────────
_rag_store             = build_rag_store()
_nutrition_agent       = NutritionKnowledgeAgent(rag_store=_rag_store)
_diet_agent            = DietRecommendationAgent()
_health_agent          = HealthAdvisoryAgent()
_food_log_agent        = FoodLogFeedbackAgent()


# ══════════════════════════════════════════════════════════════════════════════
# Health check
# ══════════════════════════════════════════════════════════════════════════════
@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "agents": ["nutrition_knowledge", "diet_recommendation", "health_advisory", "food_log_feedback"]})


# ══════════════════════════════════════════════════════════════════════════════
# Agent 1 — Nutrition Knowledge
# ══════════════════════════════════════════════════════════════════════════════
@api_bp.route("/nutrition/query", methods=["POST"])
def nutrition_query():
    """Ask any nutrition knowledge question."""
    data     = request.get_json(force=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    answer = _nutrition_agent.query(question, context=data.get("context", ""))
    return jsonify({"question": question, "answer": answer, "agent": "Nutrition Knowledge Agent"})


@api_bp.route("/nutrition/food/<food_name>", methods=["GET"])
def get_food_nutrition(food_name: str):
    """Get nutritional data for a specific food."""
    result = _nutrition_agent.get_food_nutrition(food_name)
    return jsonify(result)


@api_bp.route("/nutrition/compare", methods=["POST"])
def compare_foods():
    """Compare nutrition of two foods."""
    data  = request.get_json(force=True) or {}
    food1 = data.get("food1", "").strip()
    food2 = data.get("food2", "").strip()
    if not food1 or not food2:
        return jsonify({"error": "food1 and food2 are required"}), 400
    result = _nutrition_agent.compare_foods(food1, food2)
    return jsonify({"food1": food1, "food2": food2, "comparison": result})


@api_bp.route("/nutrition/foods", methods=["GET"])
def list_foods():
    """List all foods in the local database."""
    return jsonify({"foods": list(_nutrition_agent.FOOD_DATABASE.keys())})


# ══════════════════════════════════════════════════════════════════════════════
# Agent 2 — Diet Recommendation
# ══════════════════════════════════════════════════════════════════════════════
@api_bp.route("/diet/meal-plan", methods=["POST"])
def generate_meal_plan():
    """Generate a personalized meal plan."""
    data = request.get_json(force=True) or {}
    try:
        profile = UserProfile(
            name              = data.get("name", "User"),
            age               = int(data.get("age", 30)),
            gender            = data.get("gender", "male"),
            weight_kg         = float(data.get("weight_kg", 70)),
            height_cm         = float(data.get("height_cm", 170)),
            activity_level    = data.get("activity_level", "moderate"),
            goal              = data.get("goal", "maintain"),
            health_conditions = data.get("health_conditions", []),
            allergies         = data.get("allergies", []),
            dietary_preference= data.get("dietary_preference", "omnivore"),
            cuisine_preference= data.get("cuisine_preference", "any"),
        )
        days   = int(data.get("days", 7))
        result = _diet_agent.generate_meal_plan(profile, days=days)
        return jsonify(result)
    except Exception as exc:
        logger.error("Meal plan error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@api_bp.route("/diet/quick-plan", methods=["POST"])
def quick_meal_plan():
    """Generate a quick single-day meal plan."""
    data = request.get_json(force=True) or {}
    try:
        profile = UserProfile(
            age               = int(data.get("age", 30)),
            gender            = data.get("gender", "male"),
            weight_kg         = float(data.get("weight_kg", 70)),
            height_cm         = float(data.get("height_cm", 170)),
            activity_level    = data.get("activity_level", "moderate"),
            goal              = data.get("goal", "maintain"),
            health_conditions = data.get("health_conditions", []),
            dietary_preference= data.get("dietary_preference", "omnivore"),
        )
        result = _diet_agent.generate_single_day_plan(profile)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api_bp.route("/diet/suggest-meal", methods=["POST"])
def suggest_meal():
    """Suggest a specific meal for a given goal."""
    data      = request.get_json(force=True) or {}
    goal      = data.get("goal", "maintain weight")
    meal_type = data.get("meal_type", "lunch")
    prefs     = data.get("preferences", "")
    result    = _diet_agent.suggest_meal_for_goal(goal, meal_type, prefs)
    return jsonify({"goal": goal, "meal_type": meal_type, "suggestion": result})


# ══════════════════════════════════════════════════════════════════════════════
# Agent 3 — Health Advisory
# ══════════════════════════════════════════════════════════════════════════════
@api_bp.route("/health-advisory/condition", methods=["POST"])
def get_condition_advice():
    """Get dietary advice for a health condition."""
    data      = request.get_json(force=True) or {}
    condition = data.get("condition", "").strip()
    if not condition:
        return jsonify({"error": "condition is required"}), 400
    result = _health_agent.get_condition_advice(condition)
    return jsonify(result)


@api_bp.route("/health-advisory/preventive", methods=["POST"])
def get_preventive_plan():
    """Generate a preventive nutrition plan."""
    data         = request.get_json(force=True) or {}
    risk_factors = data.get("risk_factors", [])
    if not risk_factors:
        return jsonify({"error": "risk_factors list is required"}), 400
    plan = _health_agent.get_preventive_plan(risk_factors)
    return jsonify({"risk_factors": risk_factors, "plan": plan})


@api_bp.route("/health-advisory/assess", methods=["POST"])
def assess_diet():
    """Assess health risks in a dietary pattern."""
    data    = request.get_json(force=True) or {}
    pattern = data.get("dietary_pattern", "").strip()
    if not pattern:
        return jsonify({"error": "dietary_pattern is required"}), 400
    assessment = _health_agent.assess_diet_health_risk(pattern)
    return jsonify({"dietary_pattern": pattern, "assessment": assessment})


@api_bp.route("/health-advisory/conditions", methods=["GET"])
def list_conditions():
    """List all supported health conditions."""
    return jsonify({"conditions": _health_agent.get_all_conditions()})


# ══════════════════════════════════════════════════════════════════════════════
# Agent 4 — Food Log & Feedback
# ══════════════════════════════════════════════════════════════════════════════
@api_bp.route("/food-log/log-text", methods=["POST"])
def log_meal_text():
    """Log a meal from free-text description."""
    data      = request.get_json(force=True) or {}
    text      = data.get("text", "").strip()
    meal_type = data.get("meal_type", "meal")
    if not text:
        return jsonify({"error": "text is required"}), 400
    result = _food_log_agent.log_meal_text(text, meal_type)
    return jsonify(result)


@api_bp.route("/food-log/log", methods=["POST"])
def log_meal_structured():
    """Log a meal with structured nutritional data."""
    data = request.get_json(force=True) or {}
    try:
        result = _food_log_agent.log_meal_structured(
            food_name  = data.get("food_name", "food"),
            quantity_g = float(data.get("quantity_g", 100)),
            meal_type  = data.get("meal_type", "meal"),
            calories   = float(data.get("calories", 0)),
            protein_g  = float(data.get("protein_g", 0)),
            carbs_g    = float(data.get("carbs_g", 0)),
            fat_g      = float(data.get("fat_g", 0)),
            fiber_g    = float(data.get("fiber_g", 0)),
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api_bp.route("/food-log/water", methods=["POST"])
def log_water():
    """Log water intake in ml."""
    data = request.get_json(force=True) or {}
    ml   = float(data.get("ml", 250))
    return jsonify(_food_log_agent.log_water(ml))


@api_bp.route("/food-log/feedback", methods=["GET"])
def get_feedback():
    """Get instant nutritional feedback for today's log."""
    log_date = request.args.get("date")
    feedback = _food_log_agent.get_instant_feedback(log_date)
    return jsonify({"feedback": feedback})


@api_bp.route("/food-log/today", methods=["GET"])
def get_today_log():
    """Get today's food log."""
    return jsonify(_food_log_agent.get_daily_log())


@api_bp.route("/food-log/date/<log_date>", methods=["GET"])
def get_log_by_date(log_date: str):
    """Get food log for a specific date (YYYY-MM-DD)."""
    return jsonify(_food_log_agent.get_daily_log(log_date))


@api_bp.route("/food-log/weekly-summary", methods=["GET"])
def get_weekly_summary():
    """Get weekly nutritional summary."""
    return jsonify(_food_log_agent.get_weekly_summary())


@api_bp.route("/food-log/all", methods=["GET"])
def get_all_logs():
    """Get all food logs."""
    return jsonify(_food_log_agent.get_all_logs())


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard data endpoint
# ══════════════════════════════════════════════════════════════════════════════
@api_bp.route("/dashboard", methods=["GET"])
def get_dashboard_data():
    """Aggregate data for the visualization dashboard."""
    from config import DAILY_RECOMMENDED
    from datetime import date

    today_log = _food_log_agent.get_daily_log()
    all_logs  = _food_log_agent.get_all_logs()

    # Build time-series for weekly chart
    weekly_series = []
    for date_key in sorted(all_logs.keys())[-7:]:
        t = all_logs[date_key].get("totals", {})
        weekly_series.append({
            "date":      date_key,
            "calories":  t.get("calories", 0),
            "protein_g": t.get("protein_g", 0),
            "carbs_g":   t.get("carbs_g", 0),
            "fat_g":     t.get("fat_g", 0),
        })

    return jsonify({
        "today":        today_log,
        "weekly":       weekly_series,
        "recommended":  DAILY_RECOMMENDED,
        "food_db_size": len(_nutrition_agent.FOOD_DATABASE),
    })
