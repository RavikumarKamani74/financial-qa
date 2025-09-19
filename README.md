# Financial Document Q&A

A local Streamlit app that extracts financial metrics from Excel/PDF files and provides an interactive question-answering chat interface. Uses Ollama (optional) as a local LLM fallback.

## Features
- Upload Excel (`.xlsx`) and PDF files to extract Revenue / Expenses / Profit.
- Saves structured `*_summary.json` files to `uploads/`.
- Chat interface with deterministic numeric answers (profit margin, year queries), trend charts, and Ollama fallback for open questions.
- Polished UI: KPI cards, sparkline charts, chat bubbles, export chat JSON.

## Quick setup (Windows)
1. Create and activate a virtual environment:
```powershell
python -m venv .venv
.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app_merged.py


Open the shown local URL in your browser (default http://localhost:8501
).

Usage

In Upload & Extract tab: upload sample_income.xlsx or your PDF.

Check uploads/<filename>_summary.json is created.

Open Chat & Q&A tab: ask questions like:

What was revenue in 2022?

Show profit margin for 2023

Show revenue trend

Files

app_merged.py — merged, polished UI Streamlit app (upload + chat).

requirements.txt — dependencies.

sample_income.xlsx — sample Excel for testing.