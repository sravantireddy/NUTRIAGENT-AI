"""
Live smoke test — hits every IBM AI endpoint on the running server.
"""
import requests
import json
import time
import sys

BASE = "http://localhost:5000/api/v1"
SEP  = "=" * 56

def req(method, path, body=None, timeout=90):
    url = BASE + path
    t0  = time.time()
    if method == "GET":
        r = requests.get(url, timeout=timeout)
    else:
        r = requests.post(url, json=body, timeout=timeout)
    elapsed = round(time.time() - t0, 1)
    return r.status_code, r.json(), elapsed

def run():
    print(f"\n{SEP}")
    print("  NutriAgent AI  -  Full Live Smoke Test")
    print(f"{SEP}\n")

    passed = failed = 0

    # ── 1. Health ────────────────────────────────────────────
    status, d, t = req("GET", "/health")
    ok = status == 200 and d.get("status") == "ok"
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] Health check ({t}s)")
    print(f"       Agents: {d.get('agents')}")
    if ok: passed += 1
    else:  failed += 1

    # ── 2. Food lookup (no AI) ────────────────────────────────
    status, d, t = req("GET", "/nutrition/food/salmon")
    ok = status == 200 and "per_100g" in d
    tag = "PASS" if ok else "FAIL"
    macros = d.get("per_100g", {})
    print(f"\n[{tag}] Food lookup — salmon ({t}s)")
    print(f"       Cal={macros.get('calories')} Protein={macros.get('protein')}g Omega3={macros.get('omega3')}g")
    if ok: passed += 1
    else:  failed += 1

    # ── 3. Food DB list ───────────────────────────────────────
    status, d, t = req("GET", "/nutrition/foods")
    ok = status == 200 and len(d.get("foods", [])) > 0
    tag = "PASS" if ok else "FAIL"
    print(f"\n[{tag}] Food database ({t}s)  —  {len(d.get('foods', []))} items")
    if ok: passed += 1
    else:  failed += 1

    # ── 4. Health condition list ──────────────────────────────
    status, d, t = req("GET", "/health-advisory/conditions")
    ok = status == 200 and len(d.get("conditions", [])) >= 4
    tag = "PASS" if ok else "FAIL"
    print(f"\n[{tag}] Condition list ({t}s)  —  {d.get('conditions')}")
    if ok: passed += 1
    else:  failed += 1

    # ── 5. Log a structured meal ──────────────────────────────
    status, d, t = req("POST", "/food-log/log", {
        "food_name": "Grilled Paneer", "quantity_g": 150,
        "meal_type": "lunch", "calories": 280,
        "protein_g": 21, "carbs_g": 3, "fat_g": 20
    })
    ok = status == 200 and "entry" in d
    tag = "PASS" if ok else "FAIL"
    e   = d.get("entry", {})
    print(f"\n[{tag}] Meal log - structured ({t}s)")
    print(f"       {e.get('food_name')} | {e.get('calories')} kcal | {e.get('protein_g')}g protein")
    if ok: passed += 1
    else:  failed += 1

    # ── 6. Log water ──────────────────────────────────────────
    status, d, t = req("POST", "/food-log/water", {"ml": 500})
    ok = status == 200
    tag = "PASS" if ok else "FAIL"
    print(f"\n[{tag}] Water log ({t}s)  —  Total: {d.get('total_water_ml')}ml")
    if ok: passed += 1
    else:  failed += 1

    # ── 7. Today's log ────────────────────────────────────────
    status, d, t = req("GET", "/food-log/today")
    ok = status == 200
    tag = "PASS" if ok else "FAIL"
    tot = d.get("totals", {})
    print(f"\n[{tag}] Today's log ({t}s)")
    print(f"       {len(d.get('entries', []))} entries | {tot.get('calories', 0)} kcal | {tot.get('protein_g', 0)}g protein")
    if ok: passed += 1
    else:  failed += 1

    # ── 8. Dashboard ──────────────────────────────────────────
    status, d, t = req("GET", "/dashboard")
    ok = status == 200 and "recommended" in d
    tag = "PASS" if ok else "FAIL"
    print(f"\n[{tag}] Dashboard ({t}s)  —  food_db={d.get('food_db_size')}")
    if ok: passed += 1
    else:  failed += 1

    # ── IBM AI calls ─────────────────────────────────────────
    print(f"\n--- IBM watsonx.ai  (llama-3-3-70b-instruct) ----------")

    # 9. Nutrition query
    print("\n[....] Nutrition Knowledge Agent — querying IBM AI...")
    status, d, t = req("POST", "/nutrition/query", {
        "question": "What are the top 5 foods highest in iron for vegetarians?"
    })
    ok = status == 200 and len(d.get("answer", "")) > 50
    tag = "PASS" if ok else "FAIL"
    ans = d.get("answer", "ERROR")
    print(f"[{tag}] Nutrition Query ({t}s)")
    print(f"       {ans[:280]}{'...' if len(ans)>280 else ''}")
    if ok: passed += 1
    else:  failed += 1

    # 10. Meal plan
    print("\n[....] Diet Recommendation Agent — generating meal plan...")
    status, d, t = req("POST", "/diet/quick-plan", {
        "name": "Priya", "age": 28, "gender": "female",
        "weight_kg": 58, "height_cm": 162,
        "activity_level": "moderate", "goal": "lose",
        "dietary_preference": "vegetarian"
    })
    ok = status == 200 and "meal_plan" in d
    tag = "PASS" if ok else "FAIL"
    mt  = d.get("macro_targets", {})
    mp  = d.get("meal_plan", "ERROR")
    print(f"[{tag}] Diet Recommendation ({t}s)")
    print(f"       TDEE={d.get('tdee')} kcal | P={mt.get('protein_g')}g C={mt.get('carbs_g')}g F={mt.get('fat_g')}g")
    print(f"       {mp[:250]}{'...' if len(mp)>250 else ''}")
    if ok: passed += 1
    else:  failed += 1

    # 11. Health advisory
    print("\n[....] Health Advisory Agent — diabetes guidance...")
    status, d, t = req("POST", "/health-advisory/condition", {"condition": "diabetes"})
    ok = status == 200 and len(d.get("ai_advice", "")) > 50
    tag = "PASS" if ok else "FAIL"
    adv = d.get("ai_advice", "ERROR")
    print(f"[{tag}] Health Advisory ({t}s)")
    print(f"       {adv[:280]}{'...' if len(adv)>280 else ''}")
    if ok: passed += 1
    else:  failed += 1

    # 12. Food log feedback
    print("\n[....] Food Log Agent — AI feedback on today's intake...")
    status, d, t = req("GET", "/food-log/feedback")
    ok = status == 200 and len(d.get("feedback", "")) > 30
    tag = "PASS" if ok else "FAIL"
    fb  = d.get("feedback", "ERROR")
    print(f"[{tag}] Food Log Feedback ({t}s)")
    print(f"       {fb[:280]}{'...' if len(fb)>280 else ''}")
    if ok: passed += 1
    else:  failed += 1

    # ── Summary ───────────────────────────────────────────────
    total = passed + failed
    print(f"\n{SEP}")
    print(f"  RESULTS: {passed}/{total} passed  ({failed} failed)")
    print(f"  App URL: http://localhost:5000")
    print(f"{SEP}\n")

    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    run()
