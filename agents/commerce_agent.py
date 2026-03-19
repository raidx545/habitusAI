"""
commerce_agent.py — Agent 4: Commerce Agent
Fetches real IKEA products via Apify (pintostudio/ikea-product-search).

Pipeline:
  1. Groq → decides which furniture categories are needed (based on style + budget)
  2. Apify → fetches real IKEA products per category (runs in ~10s)
  3. USD → INR conversion + budget filtering
  4. Returns CommerceOutput with real IKEA names, prices, URLs, images

Runs on Mac. Requires APIFY_TOKEN + GROQ_API_KEY in .env
"""

import os, sys, json, asyncio, time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

try:
    from apify_client import ApifyClient
    from groq import Groq
except ImportError:
    print("[CommerceAgent] Run: pip install apify-client groq")

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path('/content/drive/MyDrive/HabitusAI')
sys.path.insert(0, str(ROOT))
from contracts import AgentInput, StyleOutput, IKEAProduct, CommerceOutput

# ── Config ────────────────────────────────────────────────────────────────────
ACTOR_ID       = "pintostudio/ikea-product-search"
USD_TO_INR     = 84          # approximate conversion rate
MAX_WAIT_SECS  = 60          # max wait for each Apify run
PRODUCTS_PER_Q = 5           # how many results to fetch per search query
LOCALE         = "en_US"     # IKEA locale (prices in USD, converted to INR)

# Style-to-search-keyword mapping for better IKEA results
STYLE_KEYWORDS = {
    "scandinavian": "light wood natural",
    "minimalist":   "white simple",
    "japandi":      "bamboo natural minimal",
    "maximalist":   "colorful pattern",
    "industrial":   "metal dark",
    "bohemian":     "rattan wicker",
    "coastal":      "white blue natural",
    "contemporary": "modern sleek",
}

# Category → IKEA search query mapping
CATEGORY_QUERIES = {
    "couch":        "sofa",
    "chair":        "armchair",
    "bed":          "bed frame",
    "dining table": "dining table",
    "tv":           "tv unit media bench",
    "potted plant": "plant pot indoor",
    "vase":         "vase",
    "book":         "bookshelf bookcase",
    "laptop":       "desk",
    "rug":          "rug",
    "lamp":         "floor lamp",
    "shelf":        "shelving unit",
    "_default":     "furniture",
}


# ── Groq: decide which categories to buy ─────────────────────────────────────
def _decide_categories(agent_input: AgentInput, style_label: str) -> list[dict]:
    """
    Uses Groq to decide which IKEA furniture categories to search for
    based on style, user intent, and budget.
    Returns a list of {category, priority, max_price_inr}.
    """
    client = Groq()
    categories_list = ", ".join(CATEGORY_QUERIES.keys())
    budget = agent_input.budget_inr or 50000

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are an interior design shopping assistant. "
                    f"Given the user's request, style, and budget of ₹{budget}, "
                    f"select 2-4 IKEA furniture categories to purchase. "
                    f"Available categories: {categories_list}. "
                    f"Return JSON: {{\"items\": ["
                    f"{{\"category\": \"...\", \"search_query\": \"...\", \"max_price_inr\": <int>}}, ...]}} "
                    f"`search_query` MUST be a simple 1-2 word query that works well on IKEA (e.g. 'white desk', 'rattan chair', 'sofa', 'rug'). "
                    f"Distribute budget roughly equally. Keep items within budget total."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Style: {style_label}\n"
                    f"Request: {agent_input.user_intent}\n"
                    f"Budget: ₹{budget}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=256,
        response_format={"type": "json_object"},
    )

    result = json.loads(resp.choices[0].message.content)
    items = result.get("items", [])
    print(f"[CommerceAgent] Groq selected categories: {[i['category'] for i in items]}")
    return items


# ── Apify: search IKEA for one query ─────────────────────────────────────────
def _search_ikea(query: str, max_price_inr: int) -> list[IKEAProduct]:
    """
    Runs the Apify IKEA actor synchronously for a single search query.
    Filters results to fit within max_price_inr.
    Returns a list of IKEAProduct.
    """
    client = ApifyClient(os.environ["APIFY_TOKEN"])
    max_price_usd = max_price_inr / USD_TO_INR

    # Use a simpler query if the first one fails
    run = client.actor(ACTOR_ID).call(
        run_input={
            "query":      query,
            "localeCode": LOCALE,
            "perPage":    PRODUCTS_PER_Q,
            "page":       1,
            "sortBy":     "RELEVANCE",
        },
        wait_secs=MAX_WAIT_SECS,
    )

    if not run or run.get("status") != "SUCCEEDED":
        print(f"[CommerceAgent] Apify run failed for query='{query}'")
        return []

    products = []
    dataset_id = run.get("defaultDatasetId")
    if not dataset_id:
        return []

    raw_items = client.dataset(dataset_id).list_items(limit=PRODUCTS_PER_Q).items
    print(f"[CommerceAgent] Apify returned {len(raw_items)} raw dataset items")

    # The actual products are nested inside the raw items
    items = []
    for raw in raw_items:
        if "results" in raw:
            items.extend(raw.get("results", {}).get("items", {}).get("products", []))
        elif "searchResultPage" in raw:
            items.extend(raw.get("searchResultPage", {}).get("products", {}).get("main", {}).get("items", []))
        else:
            items.append(raw) # fallback if they are flat

    for item in items:
        # Parse price
        price_usd = None
        sales = item.get("salesPrice", {})
        if sales:
            price_usd = sales.get("numeral")

        if price_usd is None:
            print(f"  [Skip] No price found for '{item.get('name')}'")
            continue

        if price_usd > max_price_usd:
            print(f"  [Skip] Price ${price_usd} over limit ${max_price_usd:.2f} for '{item.get('name')}'")
            continue  # over budget for this category

        price_inr = int(price_usd * USD_TO_INR)
        name      = item.get("name", "").strip()
        type_name = item.get("typeName", "").strip()

        if not name:
            continue

        url       = item.get("pipUrl", "")
        image_url = item.get("mainImageUrl", "")

        products.append(IKEAProduct(
            name=f"{name} {type_name}".strip(),
            category=type_name.lower() or "furniture",
            price_inr=price_inr,
            url=url,
            image_url=image_url,
        ))

        # Take the best matching affordable product
        if products:
            break

    print(f"[CommerceAgent] '{query}' → {len(products)} product(s)")
    return products


# ── Main Run Function ─────────────────────────────────────────────────────────
def run(
    agent_input: AgentInput,
    style_output: Optional[StyleOutput] = None,
) -> CommerceOutput:
    """
    Main entry point — called by the orchestrator.
    style_output is optional; style is extracted from user_intent if not provided.
    """
    print(f"[CommerceAgent] Budget: ₹{agent_input.budget_inr}")

    style_label = (
        style_output.style_label
        if style_output
        else "contemporary"
    )
    style_kw = STYLE_KEYWORDS.get(style_label, "")

    # Step 1: Groq decides categories + per-item budget
    categories = _decide_categories(agent_input, style_label)

    # Step 2: Fetch real IKEA products per category
    all_products: list[IKEAProduct] = []

    for item in categories:
        raw_cat    = item.get("category", "_default")
        max_price  = int(item.get("max_price_inr", (agent_input.budget_inr or 50000) // 3))
        # Use simple 1-2 word query from Groq, fallback to category name
        query      = item.get("search_query", raw_cat)

        print(f"[CommerceAgent] Searching IKEA: '{query}' (max ₹{max_price})")
        products = _search_ikea(query, max_price)
        all_products.extend(products)

    # Step 3: Compute totals
    total_price   = sum(p.price_inr for p in all_products)
    within_budget = total_price <= (agent_input.budget_inr or 50000)

    print(f"[CommerceAgent] Total: ₹{total_price:,} | Within budget: {within_budget}")
    print(f"[CommerceAgent] Products: {[p.name for p in all_products]}")

    return CommerceOutput(
        products=all_products,
        total_price_inr=total_price,
        within_budget=within_budget,
    )


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from agents.style_agent import run as style_run

    inp = AgentInput(
        image_url="https://images.unsplash.com/photo-1618220179428-22790b461013?w=1080",
        user_intent="Make it Scandinavian and cozy",
        budget_inr=45000,
    )

    style_out = style_run(inp)
    result    = run(inp, style_out)

    print(f"\n=== CommerceOutput ===")
    print(f"Products    : {len(result.products)}")
    print(f"Total       : ₹{result.total_price_inr:,}")
    print(f"Within budget: {result.within_budget}")
    for p in result.products:
        print(f"  [{p.category:12s}] {p.name:30s} ₹{p.price_inr:,}")
        print(f"             URL  : {p.url}")
        print(f"             Image: {p.image_url}")
