"""
frontend/app.py — Interactive Gradio UI
A visual interface that chains together all 4 Agents and the SDXL Pipeline.

Run using:
$ pip install gradio
$ python frontend/app.py
"""

import sys, os
from pathlib import Path
import asyncio
import gradio as gr
from PIL import Image
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Add project root so imports work
try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from contracts import AgentInput
from orchestrator.orchestrator import orchestrate
from orchestrator.aggregator import aggregate

# Try importing the SDXL Gen Pipeline (will only work on GPU machines like Colab)
try:
    from generation.sdxl_pipeline import generate_room
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    print("\n[UI] ⚠️ SDXL Pipeline dependencies missing! Generation will be skipped.")


async def process_room(image_filepath, intent, budget):
    """
    Main processing function called when the user clicks 'Redesign Room'
    """
    if not image_filepath:
        return None, "Please upload an image.", "", "", []

    print(f"\n[UI] Starting redesign for intent: '{intent}' with budget ₹{budget}")

    agent_input = AgentInput(
        image_url=image_filepath,
        user_intent=intent,
        budget_inr=int(budget)
    )

    # 1. ORCHESTRATE
    print("[UI] Running Orchestrator (4 Agents)...")
    try:
        orch_result = await orchestrate(agent_input)
    except Exception as e:
        return None, f"Orchestrator Error: {e}", "", "", []

    # 2. AGGREGATE
    print("[UI] Running Aggregator...")
    try:
        agg_result = aggregate(orch_result)
    except Exception as e:
        return None, f"Aggregator Error: {e}", "", "", []

    # 3. GENERATE (SDXL)
    if PIPELINE_AVAILABLE:
        print("[UI] Starting SDXL ControlNet Generation...")
        out_path = f"/tmp/final_design_{os.path.basename(image_filepath)}.png"
        try:
            final_img_path = generate_room(agg_result, out_path)
            final_img = Image.open(final_img_path)
        except Exception as e:
            print(f"[UI] SDXL Generation failed: {e}")
            final_img = None
    else:
        final_img = None
        print("[UI] Skipping SDXL (dependencies missing).")

    # Format products for UI Dataframe display
    product_data = []
    for p in agg_result.get("products", []):
        product_data.append([p.name, p.category, f"₹{p.price_inr}", p.url])

    summary = agg_result.get("design_summary", "No summary provided.")
    sdxl_prompt = agg_result.get("sdxl_prompt", "")
    
    style_label = orch_result["agent_outputs"]["style"].style_label
    style_conf = orch_result["agent_outputs"]["style"].confidence
    style_info = f"Style: {style_label.title()} (Confidence: {style_conf:.2f})"

    return final_img, summary, style_info, sdxl_prompt, product_data

# ── Gradio UI Layout ──────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as app:
    gr.Markdown("# 🛋️ HabitusAI: Agentic Interior Designer")
    gr.Markdown("Upload a photo of your room, describe what you want, and let our multi-agent system completely redesign it with real real-world furniture under your budget.")

    with gr.Row():
        # LEFT COLUMN (Inputs)
        with gr.Column(scale=1):
            input_image = gr.Image(type="filepath", label="Upload Room Photo", height=300)
            user_intent = gr.Textbox(lines=2, placeholder="e.g. 'Make it a cozy Scandinavian living room'", label="What do you want?")
            budget_slider = gr.Slider(minimum=5000, maximum=500000, step=1000, value=45000, label="Budget (INR)")
            
            submit_btn = gr.Button("🎨 Redesign Room", variant="primary")
            
            if not PIPELINE_AVAILABLE:
                gr.HTML("<div style='color:red; font-size:12px;'>⚠️ Running without GPU/Diffusers. Image Generation is disabled (Orchestrator only).</div>")

        # RIGHT COLUMN (Outputs)
        with gr.Column(scale=2):
            output_image = gr.Image(label="Redesigned Room (SDXL Base + ControlNet Depth)", height=400)
            
            with gr.Row():
                design_summary = gr.Textbox(label="Agent Summary", interactive=False)
                style_info = gr.Textbox(label="Detected Style", interactive=False)
            
            with gr.Accordion("Technical Details (SDXL Prompt)", open=False):
                sdxl_prompt_output = gr.Textbox(label="Synthesized SDXL Prompt", interactive=False)
                
            gr.Markdown("### 🛒 Recommended Furniture (IKEA)")
            products_table = gr.Dataframe(
                headers=["Product Name", "Category", "Price", "Link"],
                datatype=["str", "str", "str", "str"],
                col_count=(4, "fixed"),
                interactive=False
            )

    submit_btn.click(
        fn=process_room,
        inputs=[input_image, user_intent, budget_slider],
        outputs=[output_image, design_summary, style_info, sdxl_prompt_output, products_table]
    )

if __name__ == "__main__":
    print("[UI] Starting HabitusAI server...")
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
