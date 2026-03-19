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
    Main processing generator for Gradio. Yielding intermediate text updates
    keeps the Cloudflare WebSockets alive and prevents the dreaded 100-second 
    silent disconnected timeout!
    """
    if not image_filepath:
        yield None, "Please upload an image.", "", "", []
        return

    print(f"\n[UI] Starting redesign for intent: '{intent}' with budget ₹{budget}")
    
    # ── Heartbeat 1 ──
    yield None, "⏳ 1. Orchestrating Agents (Vision, Style, Layout, Commerce)...", "Waiting...", "Waiting...", []

    agent_input = AgentInput(
        image_url=image_filepath,
        user_intent=intent,
        budget_inr=int(budget)
    )

    try:
        orch_result = await orchestrate(agent_input)
    except Exception as e:
        yield None, f"❌ Orchestrator Error: {e}", "", "", []
        return

    # ── Heartbeat 2 ──
    yield None, "⏳ 2. Synthesizing SDXL Prompts & Extracted Data...", "Waiting...", "Waiting...", []
    
    try:
        agg_result = aggregate(orch_result)
    except Exception as e:
        yield None, f"❌ Aggregator Error: {e}", "", "", []
        return

    # ── Format UI Data for Heartbeat 3 ──
    product_data = []
    for p in agg_result.get("products", []):
        name = p.get("name", "Unknown") if isinstance(p, dict) else getattr(p, "name", "Unknown")
        category = p.get("category", "") if isinstance(p, dict) else getattr(p, "category", "")
        price = p.get("price_inr", 0) if isinstance(p, dict) else getattr(p, "price_inr", 0)
        url = p.get("url", "") if isinstance(p, dict) else getattr(p, "url", "")
        # Format the URL as a clickable Markdown link
        marked_url = f'<a href="{url}" target="_blank">View Product</a>' if url else "N/A"
        product_data.append([name, category, f"₹{price}", marked_url])

    style_label = orch_result["agent_outputs"]["style"].style_label
    style_conf = orch_result["agent_outputs"]["style"].confidence
    style_info = f"Style: {style_label.title()} (Confidence: {style_conf:.2f})"
    sdxl_prompt = agg_result.get("sdxl_prompt", "")
    summary = agg_result.get("design_summary", "No summary provided.")

    # ── Heartbeat 3 (The Long 2+ min Wait for SDXL) ──
    yield None, "⏳ 3. Firing up T4 GPU for SDXL ControlNet generation (takes ~2 mins)...", style_info, sdxl_prompt, product_data

    if PIPELINE_AVAILABLE:
        out_path = f"/tmp/final_design_{os.path.basename(image_filepath)}.png"
        try:
            # Releasing blocking thread correctly to allow asyncio to pump WebSockets!
            final_img_path = await asyncio.to_thread(generate_room, agg_result, out_path)
            final_img = Image.open(final_img_path)
        except Exception as e:
            print(f"[UI] SDXL Generation failed: {e}")
            final_img = None
    else:
        final_img = None
        print("[UI] Skipping SDXL (dependencies missing).")

    # ── Final Return ──
    yield final_img, summary, style_info, sdxl_prompt, product_data

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
                datatype=["str", "str", "str", "markdown"],
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
    # .queue() enables WebSockets so the browser connection NEVER times out 
    # during 2+ minute SDXL generations!
    app.queue(max_size=10).launch(server_name="0.0.0.0", server_port=7860, share=True)
