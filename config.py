"""
Central configuration for the AI Nutrition Agent.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── IBM watsonx.ai credentials ────────────────────────────────────────────────
IBM_WATSONX_API_KEY    = os.getenv("IBM_WATSONX_API_KEY",    "n1iT6yck52a5VT2PEs8c9x5dSEtbeM4iH3tJ8c59dMUk")
IBM_WATSONX_PROJECT_ID = os.getenv("IBM_WATSONX_PROJECT_ID", "d8e56306-9234-4aba-bfcd-6045778209dc")
IBM_WATSONX_URL        = os.getenv("IBM_WATSONX_URL",        "https://us-south.ml.cloud.ibm.com")

# ── Model IDs ─────────────────────────────────────────────────────────────────
# Primary: Llama 3.3 70B instruct (confirmed working in this project env)
# Fallback chain is handled automatically in ibm_client.py
GRANITE_CHAT_MODEL      = "meta-llama/llama-3-3-70b-instruct"
GRANITE_EMBED_MODEL     = "ibm/granite-embedding-278m-multilingual"

# ── Flask ─────────────────────────────────────────────────────────────────────
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "nutrition-agent-secret-2024")
FLASK_DEBUG      = os.getenv("FLASK_DEBUG", "False").lower() == "true"
PORT             = int(os.getenv("PORT", 5000))

# ── Generation parameters ─────────────────────────────────────────────────────
GENERATION_PARAMS = {
    "max_new_tokens": 800,
    "min_new_tokens": 30,
    "temperature":    0.7,
    "top_p":          0.9,
    "top_k":          50,
    "repetition_penalty": 1.1,
}

# ── Agent system prompts ──────────────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "nutrition_knowledge": (
        "You are a Nutrition Knowledge Agent specialized in food composition and nutritional data. "
        "Provide accurate calorie, macronutrient (protein, carbs, fat, fiber), and micronutrient "
        "(vitamins, minerals) information. Always cite scientific context and use standard serving sizes."
    ),
    "diet_recommendation": (
        "You are a Diet Recommendation Agent. Create personalized meal plans considering age, weight, "
        "height, health conditions, allergies, cultural food preferences, and fitness goals. "
        "Provide breakfast, lunch, dinner, and snack recommendations with portion sizes."
    ),
    "health_advisory": (
        "You are a Health Advisory Agent focused on preventive nutrition. Provide evidence-based dietary "
        "guidance for chronic disease management including diabetes, hypertension, heart disease, obesity, "
        "and metabolic syndrome. Always recommend consulting a healthcare professional for medical decisions."
    ),
    "food_log_feedback": (
        "You are a Food Log & Feedback Agent. Analyze logged meals for nutritional completeness, "
        "identify deficiencies or excesses, and provide actionable feedback. Compare intake against "
        "recommended daily values and suggest improvements for the next meal."
    ),
}

# ── Daily recommended values (adult average) ──────────────────────────────────
DAILY_RECOMMENDED = {
    "calories": 2000, "protein_g": 50,  "carbs_g": 275,
    "fat_g":    65,   "fiber_g":  28,   "sugar_g": 50,
    "sodium_mg": 2300, "potassium_mg": 4700,
    "vitamin_c_mg": 90, "calcium_mg": 1000, "iron_mg": 18,
}
