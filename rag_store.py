"""
RAG (Retrieval-Augmented Generation) store for nutrition knowledge.
Uses ChromaDB as the vector store with IBM Granite embeddings.
Falls back to a simple in-memory TF-IDF search when ChromaDB is unavailable.
"""
from __future__ import annotations
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── Rich nutrition knowledge base ─────────────────────────────────────────────
NUTRITION_KNOWLEDGE_BASE = [
    # ── Macronutrients ────────────────────────────────────────────────────────
    {
        "id": "macro_protein",
        "content": (
            "Protein is essential for muscle repair, immune function, and enzyme production. "
            "Complete proteins contain all 9 essential amino acids. Animal sources (chicken, eggs, fish, dairy) "
            "are complete. Plant sources (legumes, grains, nuts) can be combined for completeness. "
            "RDA: 0.8g/kg body weight; athletes need 1.2–2.0g/kg. "
            "High-protein foods: chicken breast (31g/100g), Greek yogurt (17g/100g), lentils (18g/100g), eggs (13g/100g)."
        ),
        "category": "macronutrients",
    },
    {
        "id": "macro_carbohydrates",
        "content": (
            "Carbohydrates are the body's primary energy source. Simple carbs (sugars) provide quick energy; "
            "complex carbs (starch, fiber) provide sustained energy and gut health benefits. "
            "Glycemic Index (GI) measures blood sugar response: low GI (<55) foods include oats, legumes, most vegetables; "
            "high GI (>70) foods include white bread, rice cakes, glucose. "
            "Fiber (soluble + insoluble) supports gut health, cholesterol management, and satiety. RDA: 275g carbs, 28g fiber."
        ),
        "category": "macronutrients",
    },
    {
        "id": "macro_fats",
        "content": (
            "Dietary fats are vital for hormone production, brain health, and fat-soluble vitamin absorption (A, D, E, K). "
            "Unsaturated fats (olive oil, avocado, nuts, fatty fish) are heart-protective. "
            "Saturated fats (butter, red meat, coconut oil) should be limited to <10% of calories. "
            "Trans fats (partially hydrogenated oils) are harmful and should be avoided entirely. "
            "Omega-3 fatty acids (EPA, DHA in fish; ALA in flaxseed, walnuts) reduce inflammation. RDA: 65g total fat."
        ),
        "category": "macronutrients",
    },

    # ── Vitamins ──────────────────────────────────────────────────────────────
    {
        "id": "vitamin_d",
        "content": (
            "Vitamin D (calciferol) is essential for calcium absorption, bone health, immune function, and mood regulation. "
            "Deficiency is extremely common globally. Sun exposure triggers synthesis in skin. "
            "Food sources: fatty fish (salmon 447 IU/100g), fortified milk, egg yolks, mushrooms. "
            "RDA: 600–800 IU/day; deficiency threshold <20 ng/mL serum. "
            "Supplementation (1000–2000 IU) often needed in northern climates or for indoor workers."
        ),
        "category": "vitamins",
    },
    {
        "id": "vitamin_c",
        "content": (
            "Vitamin C (ascorbic acid) is a potent antioxidant essential for collagen synthesis, immune function, "
            "and iron absorption (converts non-heme iron to absorbable form). "
            "Food sources: bell peppers (128mg/100g), kiwi (93mg), broccoli (89mg), oranges (53mg), strawberries (59mg). "
            "RDA: 75–90mg/day; smokers need +35mg. Megadoses >2000mg may cause GI distress. "
            "Deficiency causes scurvy (rare in developed countries)."
        ),
        "category": "vitamins",
    },
    {
        "id": "vitamin_b12",
        "content": (
            "Vitamin B12 (cobalamin) is crucial for nerve function, DNA synthesis, and red blood cell formation. "
            "Found almost exclusively in animal products: meat, fish, dairy, eggs. "
            "Vegans and vegetarians must supplement or eat fortified foods. "
            "Deficiency causes megaloblastic anemia, neurological damage (irreversible if prolonged). "
            "RDA: 2.4mcg/day. Older adults (>50) should take supplemental B12 as absorption declines. "
            "Metformin (diabetes medication) reduces B12 absorption — monitor regularly."
        ),
        "category": "vitamins",
    },

    # ── Minerals ──────────────────────────────────────────────────────────────
    {
        "id": "mineral_iron",
        "content": (
            "Iron is critical for hemoglobin synthesis and oxygen transport. "
            "Heme iron (red meat, poultry, seafood) is 2–3× more absorbable than non-heme iron (plant sources). "
            "Absorption enhancers: vitamin C (pair with plant iron). "
            "Absorption inhibitors: calcium, tannins (tea/coffee), phytates (grains/legumes). "
            "High-iron foods: liver (6.5mg/100g), lentils (3.3mg), spinach (2.7mg), tofu (2.7mg), quinoa (1.5mg). "
            "RDA: 8mg (men), 18mg (women of childbearing age), 27mg (pregnant). "
            "Deficiency: iron-deficiency anemia — most common nutritional deficiency worldwide."
        ),
        "category": "minerals",
    },
    {
        "id": "mineral_calcium",
        "content": (
            "Calcium is the most abundant mineral in the body, essential for bone and teeth strength, "
            "muscle contraction, nerve transmission, and blood clotting. "
            "Peak bone mass is built during adolescence — adequate calcium critical in teens. "
            "Food sources: dairy (milk 276mg/100g, yogurt 200mg), fortified plant milks, "
            "broccoli (47mg), almonds (264mg), sardines with bones (382mg), tofu (350mg). "
            "RDA: 1000mg/day; 1200mg for women >50 and men >70. "
            "Vitamin D is required for calcium absorption — co-deficiency is common."
        ),
        "category": "minerals",
    },

    # ── Dietary patterns ──────────────────────────────────────────────────────
    {
        "id": "diet_mediterranean",
        "content": (
            "The Mediterranean Diet is consistently ranked the healthiest dietary pattern globally. "
            "Key components: abundant plant foods (vegetables, fruits, whole grains, legumes, nuts), "
            "olive oil as primary fat, moderate fish consumption (2+/week), moderate dairy, "
            "limited red meat (<2/week), moderate red wine (optional). "
            "Health benefits: reduces cardiovascular disease risk by 30%, reduces type 2 diabetes risk, "
            "protects against cognitive decline, supports healthy weight. "
            "Evidence: Lyon Diet Heart Study, PREDIMED trial."
        ),
        "category": "dietary_patterns",
    },
    {
        "id": "diet_dash",
        "content": (
            "The DASH (Dietary Approaches to Stop Hypertension) diet was specifically designed to lower blood pressure. "
            "Key principles: high potassium (fruits, vegetables), high calcium (dairy), high magnesium, "
            "high fiber (whole grains, legumes), low sodium (<2300mg, ideally <1500mg), low saturated fat. "
            "Recommended servings: 4-5 fruits, 4-5 vegetables, 6-8 whole grains, 2-3 dairy, "
            "≤6oz lean meat, 4-5 nuts/seeds/legumes per week. "
            "Reduces systolic BP by 8-14 mmHg in hypertensive patients."
        ),
        "category": "dietary_patterns",
    },
    {
        "id": "diet_plant_based",
        "content": (
            "Plant-based diets emphasize whole plant foods and minimize or eliminate animal products. "
            "Health benefits: lower BMI, reduced heart disease risk, lower cancer risk (colorectal), "
            "better blood sugar control, lower inflammatory markers. "
            "Nutritional considerations: "
            "Vegans must supplement B12 (critical), monitor vitamin D, omega-3 (algae-based), "
            "zinc, iron, calcium, iodine. "
            "Protein combining not necessary meal-to-meal but variety over the day is important. "
            "Well-planned vegan diets are nutritionally complete for all life stages (ADA position)."
        ),
        "category": "dietary_patterns",
    },

    # ── Disease-specific nutrition ─────────────────────────────────────────────
    {
        "id": "diabetes_nutrition",
        "content": (
            "Nutritional management of Type 2 Diabetes focuses on glycemic control, weight management, "
            "and cardiovascular risk reduction. "
            "Key strategies: consistent carbohydrate intake (~45-60g/meal), choose low-GI foods, "
            "increase dietary fiber (target 35-40g/day for T2D), limit added sugars (<10% of calories), "
            "include non-starchy vegetables at every meal. "
            "The plate method: ½ non-starchy vegetables, ¼ lean protein, ¼ whole grains/starchy vegetables. "
            "Evidence-based dietary patterns for T2D: Mediterranean, low-GI, DASH, plant-based. "
            "Alcohol: moderate amounts may be acceptable but monitor for hypoglycemia with insulin/sulfonylureas."
        ),
        "category": "disease_nutrition",
    },
    {
        "id": "heart_disease_nutrition",
        "content": (
            "Cardiovascular disease prevention through diet centers on reducing LDL cholesterol, "
            "blood pressure, inflammation, and oxidative stress. "
            "American Heart Association guidelines: eat plenty of fruits and vegetables, "
            "choose whole grains, choose lean proteins, use non-tropical vegetable oils, "
            "choose minimally processed foods, minimize added sugars (women: <25g/day, men: <36g/day), "
            "reduce sodium <2300mg/day, limit alcohol. "
            "Specific foods with cardioprotective evidence: oily fish, oats (beta-glucan), "
            "nuts, olive oil, dark leafy greens, berries, legumes, dark chocolate (>70% cocoa). "
            "The PORTFOLIO diet combining cholesterol-lowering foods reduces LDL by 30%."
        ),
        "category": "disease_nutrition",
    },
    {
        "id": "weight_management_nutrition",
        "content": (
            "Evidence-based weight management prioritizes sustainable caloric deficit over extreme restriction. "
            "A 500-750 kcal/day deficit yields 0.5-1kg/week loss — the safest sustainable rate. "
            "High protein intake (1.2-1.6g/kg) preserves lean muscle mass during weight loss. "
            "Dietary fiber (>30g/day) significantly improves satiety and reduces total caloric intake. "
            "Eating patterns: time-restricted eating (16:8 intermittent fasting) shows modest benefits. "
            "Avoid very low calorie diets (<800 kcal/day) without medical supervision. "
            "Sleep (7-9 hours) and stress management are critical — poor sleep increases ghrelin (hunger hormone). "
            "Most successful long-term approach: Mediterranean diet with modest caloric reduction."
        ),
        "category": "disease_nutrition",
    },

    # ── Hydration ─────────────────────────────────────────────────────────────
    {
        "id": "hydration",
        "content": (
            "Adequate hydration is essential for virtually every physiological process: "
            "nutrient transport, temperature regulation, joint lubrication, waste excretion, and cognitive function. "
            "Daily water needs: men ~3.7L, women ~2.7L total fluid (all sources including food). "
            "Water from food contributes ~20% of daily intake. "
            "Signs of dehydration: dark urine, fatigue, headache, poor concentration. "
            "Athletes: drink 500ml 2 hours before exercise; replace 1.5× fluid lost during activity. "
            "Electrolytes (sodium, potassium, magnesium) are lost in sweat — replenish with whole foods. "
            "Plain water is optimal; limit sugary drinks, energy drinks, and excess caffeine (>400mg/day)."
        ),
        "category": "hydration",
    },

    # ── Life stages ───────────────────────────────────────────────────────────
    {
        "id": "pregnancy_nutrition",
        "content": (
            "Pregnancy nutrition supports both maternal health and fetal development. "
            "Critical nutrients: folic acid (400-800mcg before conception and first trimester — prevents neural tube defects), "
            "iron (27mg — increased need for expanded blood volume and fetal stores), "
            "calcium (1000mg — fetal bone development), "
            "DHA omega-3 (200-300mg — fetal brain development), "
            "iodine (220mcg — thyroid function and fetal neurological development), "
            "vitamin D (600 IU). "
            "Avoid: raw/undercooked meat and eggs, high-mercury fish (swordfish, shark, king mackerel), "
            "unpasteurized dairy (listeria risk), excess vitamin A (>10,000 IU). "
            "Additional calorie needs: +340 kcal/day in 2nd trimester, +450 in 3rd."
        ),
        "category": "life_stages",
    },
    {
        "id": "elderly_nutrition",
        "content": (
            "Nutritional needs shift significantly after age 65 due to reduced appetite, absorption changes, "
            "and altered metabolism. "
            "Key concerns: protein needs remain high (1.0-1.2g/kg) to prevent sarcopenia (muscle loss); "
            "calcium and vitamin D needs increase (1200mg and 800 IU respectively); "
            "B12 absorption decreases — supplementation often needed; "
            "fiber needs remain 21-28g/day for bowel health; "
            "total calorie needs decrease but nutrient density must increase. "
            "Dehydration risk increases — thirst sensation diminishes with age; "
            "aim for 8 cups water daily. "
            "Social eating and meal variety support both physical and mental health in older adults."
        ),
        "category": "life_stages",
    },
]


# ── Simple in-memory search (TF-IDF-like) ─────────────────────────────────────
class SimpleRAGStore:
    """
    Lightweight keyword-based retrieval store.
    Used when ChromaDB / vector embeddings are unavailable.
    """

    def __init__(self, documents: list[dict]):
        self._docs = documents

    def similarity_search(self, query: str, k: int = 3) -> list:
        """Return top-k documents most relevant to the query."""
        query_terms = set(re.findall(r'\w+', query.lower()))
        scored = []
        for doc in self._docs:
            text_terms = set(re.findall(r'\w+', doc["content"].lower()))
            score = len(query_terms & text_terms)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [_DocResult(d["content"]) for _, d in scored[:k]]

    def get_by_category(self, category: str) -> list[dict]:
        return [d for d in self._docs if d.get("category") == category]

    def get_categories(self) -> list[str]:
        return list({d.get("category") for d in self._docs})


class _DocResult:
    """Mimics LangChain Document interface."""
    def __init__(self, content: str):
        self.page_content = content


# ── ChromaDB vector store (optional) ──────────────────────────────────────────
def _try_build_chroma_store():
    """Attempt to build a ChromaDB-backed vector store with Granite embeddings."""
    try:
        import chromadb
        from ibm_client import get_embeddings

        client     = chromadb.Client()
        collection = client.get_or_create_collection("nutrition_knowledge")

        texts = [d["content"] for d in NUTRITION_KNOWLEDGE_BASE]
        ids   = [d["id"]      for d in NUTRITION_KNOWLEDGE_BASE]

        # Only embed if collection is empty
        if collection.count() == 0:
            vectors = get_embeddings(texts)
            collection.add(
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=[{"category": d["category"]} for d in NUTRITION_KNOWLEDGE_BASE],
            )
            logger.info("ChromaDB vector store populated with %d documents.", len(texts))

        class ChromaRAGStore:
            def similarity_search(self, query: str, k: int = 3) -> list:
                q_vec   = get_embeddings([query])[0]
                results = collection.query(query_embeddings=[q_vec], n_results=k)
                return [_DocResult(doc) for doc in results["documents"][0]]

        return ChromaRAGStore()

    except Exception as exc:
        logger.warning("ChromaDB store unavailable (%s), using simple search.", exc)
        return None


# ── Public factory ─────────────────────────────────────────────────────────────
def build_rag_store() -> SimpleRAGStore:
    """Build and return the best available RAG store."""
    chroma = _try_build_chroma_store()
    if chroma:
        return chroma
    logger.info("Using SimpleRAGStore (keyword-based retrieval).")
    return SimpleRAGStore(NUTRITION_KNOWLEDGE_BASE)
