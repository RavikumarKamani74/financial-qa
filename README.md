# 💬 Financial Document Q&A Assistant

A Streamlit-based application to process **financial documents (PDF & Excel)** and provide an **interactive question-answering system** powered by **Ollama (local LLMs)**.  
The app extracts **revenues, expenses, profits, and other financial metrics** and allows users to query them in natural language.

---

## 🚀 Features
- 📂 Upload **PDF or Excel** financial statements  
- 🔎 Automatic extraction of **key metrics** (Revenue, Expenses, Profit, etc.)  
- 💬 **Chat-based Q&A** powered by **Ollama LLMs**  
- 📊 Trend **charts & visualizations** for financial data  
- 🖥️ Clean, professional **UI built with Streamlit**  
- ✅ Runs **locally** (no cloud required)  

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/RavikumarKamani74/financial-qa.git
cd financial-qa
2. Create Virtual Environment (Optional but Recommended)
bash
Copy code
python -m venv venv
source venv/bin/activate    # On Linux/Mac
venv\Scripts\activate       # On Windows
3. Install Dependencies
bash
Copy code
pip install -r requirements.txt
4. Install & Run Ollama
Download Ollama: https://ollama.ai

Pull a model (example: gemma2):

bash
Copy code
ollama pull gemma2
▶️ Usage
Step 1: Run Document Extractor
bash
Copy code
streamlit run app.py
Upload Excel or PDF financial documents.

The app will extract metrics and save a structured JSON summary in uploads/.

Step 2: Run Q&A Chat Interface
bash
Copy code
streamlit run app_chat.py
Loads the extracted summary.

Ask questions like:

"What was revenue in 2022?"

"Show profit margin for 2023"

"Summarize financial performance"

📸 Screenshot
<img width="1536" height="1024" alt="financial-qa" src="https://github.com/user-attachments/assets/e4dd3062-c85e-4ef8-a9ed-6da44c1d7ac8" />


📂 Project Structure
bash
Copy code
financial-qa/
│── app.py               # Document extractor (upload & parse)
│── app_chat.py          # Q&A chat interface
│── app_merged.py        # (Optional) Combined app
│── test_ollama.py       # Quick test for Ollama
│── inspect_summary.py   # Debug utility for summaries
│── requirements.txt     # Python dependencies
│── uploads/             # Uploaded files & summaries
│── README.md            # Project documentation
│── .gitignore
📝 Notes
Works with Income Statements, Balance Sheets, Cash Flow Statements

Supports conversational follow-ups in Q&A

If metrics cannot be found, LLM fallback answers are used

🎯 Success Criteria
✅ Upload financial documents
✅ Extract & preview structured data
✅ Ask & answer financial questions
✅ Display trend charts and metrics cleanly

👨‍💻 Developed as an assignment project by Ravikumar Kamani
