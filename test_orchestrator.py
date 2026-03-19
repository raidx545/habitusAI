"""
test_orchestrator.py — End-to-end test for the orchestrator + aggregator
Uses mock agent outputs so you don't need any real agents built yet.

Usage:
    source venv/bin/activate
    export GROQ_API_KEY=your_key_here
    python test_orchestrator.py
"""

import asyncio
import json
import os
import sys

# Make sure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from mocks import MOCK_INPUT, get_mock_agent_outputs
from orchestrator.orchestrator import parse_intent
from orchestrator.aggregator import aggregate


def test_parse_intent():
    """Test 1: Groq intent parsing"""
    print("\n" + "="*60)
    print("TEST 1: parse_intent() — Groq intent parsing")
    print("="*60)
    print(f"Input: '{MOCK_INPUT.user_intent}'")
    print(f"Budget: ₹{MOCK_INPUT.budget_inr:,}")

    parsed = parse_intent(MOCK_INPUT)

    print("\n✅ Groq response:")
    print(json.dumps(parsed, indent=2))

    # Validate required keys
    required_keys = ["parsed_intent", "style_hint", "budget_inr", "priority_rooms", "extra_constraints"]
    for key in required_keys:
        assert key in parsed, f"❌ Missing key: {key}"
        print(f"  ✓ '{key}' present")

    return parsed


def test_aggregator(parsed_intent: dict):
    """Test 2: Groq aggregator — SDXL prompt generation"""
    print("\n" + "="*60)
    print("TEST 2: aggregate() — SDXL prompt generation")
    print("="*60)

    # Build a mock orchestrator_output dict (as if orchestrate() returned it)
    mock_orchestrator_output = {
        "parsed_intent": parsed_intent,
        "agent_outputs": get_mock_agent_outputs(),
    }

    result = aggregate(mock_orchestrator_output)

    print("\n✅ Aggregator output:")
    print(f"\n[SDXL PROMPT]\n{result['sdxl_prompt']}")
    print(f"\n[NEGATIVE PROMPT]\n{result['negative_prompt']}")
    print(f"\n[DESIGN SUMMARY]\n{result['design_summary']}")
    print(f"\n[DEPTH MAP PATH] {result['depth_map_path']}")
    print(f"[FLOOR PLAN PATH] {result['floor_plan_path']}")
    print(f"[WITHIN BUDGET] {result['within_budget']}")
    print(f"[TOTAL PRICE] ₹{result['total_price_inr']:,}")
    print(f"[PRODUCTS] {len(result['products'])} items")

    # Validate required keys
    required_keys = ["sdxl_prompt", "negative_prompt", "design_summary",
                     "depth_map_path", "floor_plan_path", "products"]
    for key in required_keys:
        assert key in result, f"❌ Missing key: {key}"
        print(f"  ✓ '{key}' present")

    return result


if __name__ == "__main__":
    print("🚀 Interior AI — Orchestrator Pipeline Test")
    print(f"   Using model: llama-3.3-70b-versatile (Groq)")

    if not os.getenv("GROQ_API_KEY"):
        print("\n❌ GROQ_API_KEY not set. Run: export GROQ_API_KEY=your_key_here")
        sys.exit(1)

    try:
        parsed = test_parse_intent()
        result = test_aggregator(parsed)
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED — Orchestrator pipeline is working!")
        print("="*60)
    except AssertionError as e:
        print(f"\n❌ Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
