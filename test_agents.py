# -*- coding: utf-8 -*-
"""
Standalone CLI for testing all four agents without starting Flask.
Run: python test_agents.py
"""
from __future__ import annotations
import json
import sys
import os

# Force UTF-8 on Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Test runner ────────────────────────────────────────────────────────────────
PASS = "[PASS]"
FAIL = "[FAIL]"
results = []

def test(name: str, fn):
    try:
        result = fn()
        print(f"{PASS} {name}")
        if result:
            preview = str(result)[:200].replace('\n', ' ')
            print(f"   -> {preview}{'...' if len(str(result))>200 else ''}")
        results.append((name, True))
    except Exception as exc:
        print(f"{FAIL} {name}: {exc}")
        results.append((name, False))

# ===============================================================================
print("\n" + "="*60)
print("  NutriAgent AI -- Agent Test Suite")
print("  IBM Granite 3.1 · watsonx.ai")
print("="*60 + "\n")

# ── RAG Store ─────────────────────────────────────────────────────────────────
print("-- RAG Store -------------------------------------------------------")
from rag_store import build_rag_store, NUTRITION_KNOWLEDGE_BASE

def t_rag_build():
    store = build_rag_store()
    return f"Store built with {len(NUTRITION_KNOWLEDGE_BASE)} documents"

def t_rag_search():
    store = build_rag_store()
    results = store.similarity_search("protein intake for muscle building", k=2)
    assert results, "No results returned"
    return results[0].page_content[:100]

test("Build RAG store", t_rag_build)
test("RAG similarity search", t_rag_search)

# ── Agent 1: Nutrition Knowledge ──────────────────────────────────────────────
print("\n-- Agent 1: Nutrition Knowledge ------------------------------------")
from agents import NutritionKnowledgeAgent
from rag_store import build_rag_store as _rag

rag = _rag()
nk  = NutritionKnowledgeAgent(rag_store=rag)

def t_food_lookup():
    data = nk.get_food_nutrition("salmon")
    assert "per_100g" in data, "Missing per_100g"
    return data

def t_food_lookup_unknown():
    data = nk.get_food_nutrition("mystery_food_xyz")
    return data.get("message", "ok")

def t_food_compare():
    r = nk.compare_foods("oats", "quinoa")
    assert r, "Empty comparison"
    return r[:150]

def t_nutrition_query():
    r = nk.query("What are the health benefits of omega-3 fatty acids?")
    assert r, "Empty response"
    return r[:150]

test("Food lookup -- known food (salmon)", t_food_lookup)
test("Food lookup -- unknown food (graceful fallback)", t_food_lookup_unknown)
test("Food comparison (oats vs quinoa)", t_food_compare)
test("Nutrition query (omega-3 benefits)", t_nutrition_query)

# ── Agent 2: Diet Recommendation ─────────────────────────────────────────────
print("\n-- Agent 2: Diet Recommendation ------------------------------------")
from agents import DietRecommendationAgent
from agents.diet_recommendation_agent import UserProfile

dr = DietRecommendationAgent()

def t_tdee_calculation():
    profile = UserProfile(age=30, gender="female", weight_kg=65, height_cm=165, activity_level="moderate", goal="lose")
    tdee = dr._calculate_tdee(profile)
    assert 1000 < tdee < 3000, f"TDEE {tdee} out of expected range"
    return f"TDEE = {tdee:.0f} kcal"

def t_macro_targets():
    profile = UserProfile(age=25, gender="male", weight_kg=80, height_cm=180, activity_level="active", goal="build_muscle")
    tdee   = dr._calculate_tdee(profile)
    macros = dr._calculate_macros(profile, tdee)
    assert macros["protein_g"] > 0, "Protein must be > 0"
    return f"P:{macros['protein_g']}g C:{macros['carbs_g']}g F:{macros['fat_g']}g"

def t_restrictions():
    profile = UserProfile(dietary_preference="vegan", health_conditions=["diabetes"], allergies=["nuts"])
    r = dr._build_restrictions(profile)
    assert "meat" in r, "Vegan profile should restrict meat"
    return f"{len(r)} restrictions: {r[:4]}"

def t_single_day_plan():
    profile = UserProfile(age=35, goal="maintain", dietary_preference="vegetarian")
    result  = dr.generate_single_day_plan(profile)
    assert "meal_plan" in result, "Missing meal_plan key"
    return result["meal_plan"][:150]

test("TDEE calculation (female, moderate, lose)", t_tdee_calculation)
test("Macro targets (male, active, build_muscle)", t_macro_targets)
test("Restrictions builder (vegan + diabetes + nut allergy)", t_restrictions)
test("Single-day meal plan generation", t_single_day_plan)

# ── Agent 3: Health Advisory ──────────────────────────────────────────────────
print("\n-- Agent 3: Health Advisory -----------------------------------------")
from agents import HealthAdvisoryAgent

ha = HealthAdvisoryAgent()

def t_condition_match():
    key = ha._match_condition("type 2 diabetes")
    assert key == "diabetes", f"Expected 'diabetes', got '{key}'"
    return f"Matched: {key}"

def t_condition_match_alias():
    key = ha._match_condition("high blood pressure")
    assert key == "hypertension", f"Expected 'hypertension', got '{key}'"
    return f"Matched alias: {key}"

def t_condition_advice():
    result = ha.get_condition_advice("diabetes")
    assert "ai_advice" in result, "Missing ai_advice"
    return result["ai_advice"][:150]

def t_list_conditions():
    conditions = ha.get_all_conditions()
    assert len(conditions) >= 4, f"Expected 4+ conditions, got {len(conditions)}"
    return str(conditions)

test("Condition matching -- exact (diabetes)", t_condition_match)
test("Condition matching -- alias (high blood pressure)", t_condition_match_alias)
test("Condition-specific dietary advice (diabetes)", t_condition_advice)
test("List all supported conditions", t_list_conditions)

# ── Agent 4: Food Log & Feedback ──────────────────────────────────────────────
print("\n-- Agent 4: Food Log & Feedback -------------------------------------")
from agents import FoodLogFeedbackAgent

fl = FoodLogFeedbackAgent()

def t_log_structured():
    result = fl.log_meal_structured("Grilled Chicken", 150, "lunch", calories=248, protein_g=46.5, carbs_g=0, fat_g=5.4)
    assert "entry" in result, "Missing entry"
    return result["entry"]

def t_log_water():
    result = fl.log_water(500)
    assert result["water_logged_ml"] == 500
    return f"Water: {result['total_water_ml']}ml logged"

def t_daily_totals():
    result = fl.get_daily_log()
    assert "entries" in result
    return f"{len(result['entries'])} entries, totals: {result.get('totals',{})}"

def t_feedback():
    feedback = fl.get_instant_feedback()
    assert feedback, "Empty feedback"
    return feedback[:150]

def t_log_text():
    result = fl.log_meal_text("I had a bowl of oatmeal with blueberries and a glass of milk", "breakfast")
    assert "logged_entries" in result or "feedback" in result
    return str(result.get("logged_entries",""))[:100]

def t_nutrition_estimation():
    from agents.food_log_feedback_agent import FoodEntry
    entry = FoodEntry(food_name="chicken", quantity_g=200, meal_type="dinner")
    fl._estimate_nutrition(entry)
    assert entry.calories > 0, "Calories should be > 0"
    return f"200g chicken -> {entry.calories} kcal, {entry.protein_g}g protein"

test("Structured meal logging", t_log_structured)
test("Water intake logging", t_log_water)
test("Daily log totals", t_daily_totals)
test("Nutritional feedback generation", t_feedback)
test("Free-text meal logging + AI parsing", t_log_text)
test("Auto-nutrition estimation", t_nutrition_estimation)

# ── IBM Client ────────────────────────────────────────────────────────────────
print("\n-- IBM Client (watsonx.ai) ------------------------------------------")
from ibm_client import generate_text, _fallback_response, _fallback_embeddings

def t_fallback_text():
    r = _fallback_response("What are good high protein foods?", "")
    assert r, "Empty fallback response"
    return r[:100]

def t_fallback_embeddings():
    vecs = _fallback_embeddings(["test text", "another text"])
    assert len(vecs) == 2, "Expected 2 embeddings"
    assert len(vecs[0]) == 384, "Expected 384-dim embeddings"
    return f"Embeddings: 2x{len(vecs[0])} dim"

def t_generate_text():
    r = generate_text("What vitamins are in broccoli?",
                      system_prompt="You are a nutrition expert.")
    assert r, "Empty response"
    return r[:150]

test("Fallback text response (offline)", t_fallback_text)
test("Fallback embeddings (offline)", t_fallback_embeddings)
test("Generate text (IBM Granite or fallback)", t_generate_text)

# ── Summary ───────────────────────────────────────────────────────────────────
total  = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed

print("\n" + "="*60)
print(f"  Results: {passed}/{total} passed  ({failed} failed)")
print("="*60)

if failed:
    print("\nFailed tests:")
    for name, ok in results:
        if not ok:
            print(f"  {FAIL} {name}")
    sys.exit(1)
else:
    print("\n[OK] All tests passed! NutriAgent AI is ready.\n")
    sys.exit(0)
