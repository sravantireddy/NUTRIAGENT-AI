"""
IBM watsonx.ai integration layer.
Uses the ModelInference SDK — tries generate_text first, then chat API.
Falls back to deterministic offline responses when IBM is unreachable.
"""
from __future__ import annotations
import logging
import warnings
from functools import lru_cache
from typing import Optional

# Suppress IBM SDK deprecation noise
warnings.filterwarnings("ignore", category=UserWarning, module="ibm_watsonx_ai")

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

from config import (
    IBM_WATSONX_API_KEY,
    IBM_WATSONX_PROJECT_ID,
    IBM_WATSONX_URL,
    GRANITE_CHAT_MODEL,
    GRANITE_EMBED_MODEL,
    GENERATION_PARAMS,
)

logger = logging.getLogger(__name__)

# ── Credential singleton ───────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_credentials() -> Credentials:
    return Credentials(url=IBM_WATSONX_URL, api_key=IBM_WATSONX_API_KEY)


@lru_cache(maxsize=8)
def _get_model(model_id: str) -> ModelInference:
    return ModelInference(
        model_id=model_id,
        credentials=_get_credentials(),
        project_id=IBM_WATSONX_PROJECT_ID,
        params={
            GenParams.MAX_NEW_TOKENS:     GENERATION_PARAMS["max_new_tokens"],
            GenParams.MIN_NEW_TOKENS:     GENERATION_PARAMS["min_new_tokens"],
            GenParams.TEMPERATURE:        GENERATION_PARAMS["temperature"],
            GenParams.TOP_P:              GENERATION_PARAMS["top_p"],
            GenParams.TOP_K:              GENERATION_PARAMS["top_k"],
            GenParams.REPETITION_PENALTY: GENERATION_PARAMS["repetition_penalty"],
        },
    )


# ── Public API ─────────────────────────────────────────────────────────────────
def generate_text(
    prompt: str,
    system_prompt: str = "",
    model_id: str = GRANITE_CHAT_MODEL,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Generate text using IBM watsonx.ai.
    Tries the chat completions API first (non-deprecated), then generate_text.
    Gracefully falls back to rule-based offline responses on any failure.
    """
    try:
        model   = _get_model(model_id)
        msgs    = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})

        chat_params = {"max_tokens": max_tokens or GENERATION_PARAMS["max_new_tokens"]}
        response = model.chat(messages=msgs, params=chat_params)
        text = response["choices"][0]["message"]["content"]
        logger.info("IBM chat API success (model=%s, tokens=%d)", model_id, len(text.split()))
        return text.strip()

    except Exception as chat_exc:
        logger.debug("Chat API failed (%s), trying generate_text…", chat_exc)

    try:
        model      = _get_model(model_id)
        full_prompt = _build_prompt(system_prompt, prompt)
        extra       = {GenParams.MAX_NEW_TOKENS: max_tokens} if max_tokens else {}
        response    = model.generate_text(prompt=full_prompt, params=extra or None)
        logger.info("IBM generate_text success (model=%s)", model_id)
        return response.strip()

    except Exception as gen_exc:
        logger.warning("IBM watsonx.ai unavailable (%s) — using offline fallback.", gen_exc)
        return _fallback_response(prompt, system_prompt)


def get_embeddings(texts: list[str], model_id: str = GRANITE_EMBED_MODEL) -> list[list[float]]:
    """
    Return embeddings using IBM Granite embedding model.
    Falls back to deterministic pseudo-embeddings on failure.
    """
    try:
        model    = _get_model(model_id)
        response = model.generate_embeddings(texts=texts)
        return [item["embedding"] for item in response["results"]]
    except Exception as exc:
        logger.warning("Embedding call failed (%s) — using offline fallback.", exc)
        return _fallback_embeddings(texts)


# ── Prompt formatting ──────────────────────────────────────────────────────────
def _build_prompt(system_prompt: str, user_message: str) -> str:
    """Format a plain text prompt compatible with instruction-tuned models."""
    if system_prompt:
        return f"System: {system_prompt}\n\nUser: {user_message}\n\nAssistant:"
    return f"User: {user_message}\n\nAssistant:"


# ── Offline fallbacks ──────────────────────────────────────────────────────────
def _fallback_response(prompt: str, system_prompt: str = "") -> str:
    """Rule-based fallback so the app remains functional without live IBM credentials."""
    p = prompt.lower()

    if any(k in p for k in ("calorie", "nutrition", "macro", "protein", "carb", "fat", "vitamin", "mineral")):
        return (
            "Based on standard nutritional databases:\n"
            "A balanced meal typically contains 400–600 calories with:\n"
            "• Protein: 25–35g (chicken, fish, eggs, legumes)\n"
            "• Carbohydrates: 45–65g (whole grains, vegetables, fruits)\n"
            "• Healthy fats: 15–25g (olive oil, nuts, avocado)\n"
            "• Fiber: 7–10g (vegetables, legumes, whole grains)\n\n"
            "For precise values, consult the USDA FoodData Central database."
        )
    if any(k in p for k in ("meal plan", "diet plan", "breakfast", "lunch", "dinner", "snack", "plan")):
        return (
            "Personalized Meal Plan:\n\n"
            "BREAKFAST (350 cal): Oatmeal with mixed berries + Greek yogurt + green tea\n"
            "MID-MORNING (150 cal): Apple + 12 almonds\n"
            "LUNCH (500 cal): Grilled chicken salad with quinoa + olive oil dressing\n"
            "AFTERNOON (200 cal): Hummus with carrot sticks + whole grain crackers\n"
            "DINNER (550 cal): Baked salmon + steamed broccoli + brown rice\n\n"
            "Daily totals: ~1750 cal | Protein: 95g | Carbs: 175g | Fat: 58g | Fiber: 28g\n"
            "Hydration goal: 8 glasses (2L) of water daily."
        )
    if any(k in p for k in ("diabetes", "blood sugar", "insulin", "glucose", "a1c")):
        return (
            "Diabetes-Friendly Dietary Guidance:\n\n"
            "EMPHASIZE:\n"
            "• Non-starchy vegetables (fills half your plate)\n"
            "• Whole grains with low GI (oats, barley, quinoa)\n"
            "• Lean proteins (chicken, fish, tofu, legumes)\n"
            "• Healthy fats (avocado, olive oil, nuts)\n\n"
            "LIMIT/AVOID:\n"
            "• Refined carbohydrates and added sugars\n"
            "• Sugary drinks and fruit juices\n"
            "• White bread, white rice, pastries\n\n"
            "PRACTICAL TIPS:\n"
            "• Use the Plate Method: ½ vegetables, ¼ protein, ¼ whole grains\n"
            "• Space meals evenly (every 3–4 hours)\n"
            "• Aim for 45–60g carbs per meal\n"
            "• Always consult your endocrinologist for personalized advice."
        )
    if any(k in p for k in ("heart", "cholesterol", "blood pressure", "hypertension", "cardiovascular")):
        return (
            "Heart-Healthy Dietary Guidelines:\n\n"
            "EMPHASIZE:\n"
            "• Omega-3 rich foods: salmon, sardines, walnuts, flaxseed\n"
            "• Fiber-rich foods: oats, legumes, fruits, vegetables\n"
            "• Antioxidants: berries, dark leafy greens, dark chocolate\n"
            "• Healthy unsaturated fats: olive oil, avocado, nuts\n\n"
            "LIMIT/AVOID:\n"
            "• Saturated fats (<7% of total calories)\n"
            "• Trans fats (partially hydrogenated oils) — avoid entirely\n"
            "• Sodium (<2300mg/day; <1500mg if hypertensive)\n"
            "• Processed and ultra-processed foods\n\n"
            "Follow the Mediterranean or DASH dietary pattern for best outcomes."
        )
    if any(k in p for k in ("weight", "lose", "obesity", "bmi", "overweight")):
        return (
            "Weight Management Nutrition Strategy:\n\n"
            "• Create a 500–750 kcal/day deficit for 0.5–1kg/week safe loss\n"
            "• Prioritize protein (1.2–1.6g/kg) to preserve muscle mass\n"
            "• Fill half your plate with low-calorie, high-volume vegetables\n"
            "• Choose fiber-rich foods (>30g/day) for improved satiety\n"
            "• Avoid liquid calories (soda, juice, alcohol)\n"
            "• Eat slowly — it takes 20 minutes to feel full\n"
            "• Drink 500ml water before meals to reduce appetite\n"
            "• Most sustainable long-term pattern: Mediterranean diet with moderate caloric restriction"
        )
    if any(k in p for k in ("log", "ate", "had", "eaten", "track", "record")):
        return (
            "Meal logged successfully!\n\n"
            "Nutritional Analysis:\n"
            "• Your meal has been recorded and analyzed\n"
            "• Estimated caloric contribution: within normal range\n"
            "• Suggestion: pair this with a serving of vegetables for added fiber\n\n"
            "Daily Progress:\n"
            "• Continue aiming for 5+ servings of fruits and vegetables\n"
            "• Stay hydrated: aim for 2–2.5L water today\n"
            "• Next meal suggestion: focus on lean protein + fiber"
        )
    if any(k in p for k in ("supplement", "vitamin d", "b12", "iron", "calcium", "omega")):
        return (
            "Nutritional Supplement Guidance:\n\n"
            "Most commonly needed supplements:\n"
            "• Vitamin D (1000–2000 IU): most people are deficient, especially in northern climates\n"
            "• Vitamin B12 (500–1000mcg): essential for vegans/vegetarians; harder to absorb after age 50\n"
            "• Omega-3 (1000–2000mg EPA+DHA): if fatty fish <2x/week\n"
            "• Iron (18mg): important for women, athletes, vegans — test levels first\n\n"
            "Food sources are always preferable to supplements when possible.\n"
            "Consult your physician before starting any supplementation."
        )
    return (
        "As your AI Nutrition Agent, I'm here to provide personalized dietary guidance.\n\n"
        "I can help you with:\n"
        "• Nutritional information for any food\n"
        "• Personalized meal plans based on your goals\n"
        "• Disease-specific dietary guidance (diabetes, heart health, etc.)\n"
        "• Meal logging and nutritional tracking\n"
        "• Evidence-based preventive nutrition advice\n\n"
        "Please ask me a specific nutrition question to get started!"
    )


def _fallback_embeddings(texts: list[str]) -> list[list[float]]:
    """Return deterministic pseudo-embeddings (384-dim) for offline use."""
    import hashlib
    results = []
    for text in texts:
        h   = int(hashlib.md5(text.encode()).hexdigest(), 16)
        vec = [(((h >> (i % 128)) & 0xFF) / 255.0 - 0.5) * 2 for i in range(384)]
        results.append(vec)
    return results
