# habitusAI

## 🚀 Quickstart: Run the Web App (Gradio)

The easiest way to use HabitusAI is via the interactive Web UI. 

**1. On your Local Mac (for testing)**
```bash
pip install -r requirements.txt
pip install gradio
python frontend/app.py
```
*(Note: SDXL Image Generation will skip safely on Mac since it lacks a Colab T4 GPU, but all 4 Agents will run and return IKEA products + Prompts).*

**2. On Google Colab (Full Pipeline with SDXL Image Gen)**
1. Upload the entire `HabitusAI` folder to your Google Drive.
2. Open a Colab Notebook with a T4 GPU.
3. Run the following cell:
```python
!pip install -r /content/drive/MyDrive/HabitusAI/requirements.txt
!pip install diffusers transformers accelerate invisible_watermark safetensors gradio -q
import sys
sys.path.insert(0, '/content/drive/MyDrive/HabitusAI')
!python /content/drive/MyDrive/HabitusAI/frontend/app.py
```
4. Click the public `gradio.live` link to open your AI Interior Designer!

## Fast API Backend (Headless)
Run the backend API directly:
```bash
uvicorn api.main:app --reload --port 8000
```
POST to `http://localhost:8000/design` with:
```json
{
  "image_url": "https://example.com/room.jpg",
  "user_intent": "Make it scandinavian",
  "budget_inr": 45000
}
```

## System ArchitecturesAI
