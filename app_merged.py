import os
import io
import json
import re
import streamlit as st
import pandas as pd

# Optional features
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except Exception:
    PDFPLUMBER_AVAILABLE = False

try:
    from ollama import chat as ollama_chat
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

# --------------------------
# Config & directories
# --------------------------
st.set_page_config(page_title="Financial Document Q&A", layout="wide")
APP_UPLOAD_DIR = "uploads"
os.makedirs(APP_UPLOAD_DIR, exist_ok=True)
DEFAULT_MODEL = "gemma2:2b"

# --------------------------
# Utility functions
# --------------------------
def save_uploaded_file(uploaded_file) -> str:
    path = os.path.join(APP_UPLOAD_DIR, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def parse_number_like(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        s = s.replace(',', '')
        if s.startswith('(') and s.endswith(')'):
            s = '-' + s[1:-1]
        s = re.sub(r'[^\d\.\-]', '', s)
        if s == '':
            return None
        return float(s)
    except Exception:
        return None

def extract_excel_metrics(file_path):
    summary = {"source": os.path.basename(file_path), "type": "excel", "sheets": {}, "metrics": {}}
    try:
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            summary["sheets"][sheet_name] = {"rows": len(df), "cols": list(df.columns)}
            for c in df.columns:
                lc = str(c).lower()
                if any(k in lc for k in ["revenue", "sales", "net sales", "total revenue"]):
                    series_clean = [parse_number_like(v) for v in df[c].fillna("").tolist()]
                    summary["metrics"]["revenue"] = {str(i): v for i, v in enumerate(series_clean) if v is not None}
                if any(k in lc for k in ["expense", "cost", "cogs"]):
                    series_clean = [parse_number_like(v) for v in df[c].fillna("").tolist()]
                    summary["metrics"]["expenses"] = {str(i): v for i, v in enumerate(series_clean) if v is not None}
                if any(k in lc for k in ["profit", "net income", "earnings"]):
                    series_clean = [parse_number_like(v) for v in df[c].fillna("").tolist()]
                    summary["metrics"]["profit"] = {str(i): v for i, v in enumerate(series_clean) if v is not None}
        return summary
    except Exception as e:
        st.error(f"Excel parsing error: {e}")
        return summary

def extract_pdf_metrics(file_path):
    summary = {"source": os.path.basename(file_path), "type": "pdf", "text_preview": None, "tables": 0, "metrics": {}}
    if not PDFPLUMBER_AVAILABLE:
        st.warning("pdfplumber not installed; PDF parsing skipped.")
        return summary
    try:
        text_parts = []
        tables_found = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                text_parts.append(txt)
                try:
                    page_tables = page.extract_tables()
                    for t in page_tables:
                        df = pd.DataFrame(t)
                        df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
                        if not df.empty:
                            tables_found.append(df)
                except Exception:
                    pass
        summary["text_preview"] = "\n\n".join(text_parts)[:3000]
        summary["tables"] = len(tables_found)
        for df in tables_found:
            try:
                first_col = df.columns[0]
                for i, val in enumerate(df[first_col].astype(str).tolist()):
                    label = val.lower()
                    if any(k in label for k in ["revenue", "sales"]):
                        row = df.iloc[i].tolist()
                        nums = [parse_number_like(x) for x in row if parse_number_like(x) is not None]
                        summary["metrics"]["revenue"] = {str(i): v for i, v in enumerate(nums)}
                    if any(k in label for k in ["expense", "cost", "cogs"]):
                        row = df.iloc[i].tolist()
                        nums = [parse_number_like(x) for x in row if parse_number_like(x) is not None]
                        summary["metrics"]["expenses"] = {str(i): v for i, v in enumerate(nums)}
                    if any(k in label for k in ["profit", "net income", "earnings"]):
                        row = df.iloc[i].tolist()
                        nums = [parse_number_like(x) for x in row if parse_number_like(x) is not None]
                        summary["metrics"]["profit"] = {str(i): v for i, v in enumerate(nums)}
            except Exception:
                continue
        # fallback: search text lines for numbers near metric keywords
        lines = summary["text_preview"].splitlines()
        for line in lines:
            lc = line.lower()
            for metric in ["revenue", "profit", "expenses"]:
                if metric in lc:
                    nums = re.findall(r'[\d,]+(?:\.\d+)?', line)
                    if nums:
                        nums_clean = [parse_number_like(n) for n in nums]
                        summary["metrics"].setdefault(metric, {str(i): v for i, v in enumerate(nums_clean) if v is not None})
        return summary
    except Exception as e:
        st.error(f"PDF parsing error: {e}")
        return summary

def save_summary_json(summary: dict, filename_prefix: str) -> str:
    out_path = os.path.join(APP_UPLOAD_DIR, f"{filename_prefix}_summary.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return out_path

def load_latest_summary() -> dict:
    files = sorted([f for f in os.listdir(APP_UPLOAD_DIR) if f.endswith("_summary.json")])
    if not files:
        return {}
    path = os.path.join(APP_UPLOAD_DIR, files[-1])
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}

def get_metric_series(metric_raw):
    vals = []
    if isinstance(metric_raw, dict):
        try:
            items = sorted(metric_raw.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else x[0])
        except Exception:
            items = list(metric_raw.items())
        for _, v in items:
            try:
                vals.append(float(v))
            except Exception:
                pass
    elif isinstance(metric_raw, (list, tuple, pd.Series)):
        for v in metric_raw:
            try:
                vals.append(float(v))
            except Exception:
                pass
    else:
        try:
            vals = [float(metric_raw)]
        except Exception:
            vals = []
    return vals

def detect_year_labels_from_source(summary):
    src = summary.get("source")
    if not src:
        return None
    path = os.path.join(APP_UPLOAD_DIR, src)
    if not os.path.exists(path):
        files = [f for f in os.listdir(APP_UPLOAD_DIR) if f.lower().endswith(('.xls', '.xlsx'))]
        if not files:
            return None
        path = os.path.join(APP_UPLOAD_DIR, files[0])
    try:
        df = pd.read_excel(path, sheet_name=0)
    except Exception:
        return None
    year_col = None
    for col in df.columns:
        c = str(col).lower()
        if "year" in c or "period" in c or "date" in c:
            year_col = col
            break
    if year_col is None:
        for col in df.columns:
            sample = df[col].astype(str).dropna().astype(str).head(10).tolist()
            if any([s.strip().isdigit() and len(s.strip()) == 4 for s in sample]):
                year_col = col
                break
    if year_col is None:
        return None
    def extract_year(s):
        m = re.search(r'(19\d{2}|20\d{2})', str(s))
        return m.group(1) if m else str(s).strip()
    try:
        labels = [extract_year(x) for x in df[year_col].astype(str).tolist()]
        return labels
    except Exception:
        return None

def value_for_metric_year(metric, year_label, metrics_dict, year_labels):
    raw = metrics_dict.get(metric)
    if raw is None:
        return None
    vals = get_metric_series(raw)
    labels = year_labels if year_labels and len(year_labels) >= len(vals) else [str(i) for i in range(len(vals))]
    try:
        idx = labels.index(str(year_label))
        return vals[idx]
    except Exception:
        if isinstance(raw, dict) and year_label in raw:
            try:
                return float(raw[year_label])
            except Exception:
                return None
        return None

def deterministic_answer(question, summary, year_labels):
    q = question.lower()
    yrs = re.findall(r'(19\d{2}|20\d{2})', q)
    metrics = summary.get("metrics", {})
    if ('profit margin' in q or ('margin' in q and 'profit' in q)) and yrs:
        y = yrs[0]
        p = value_for_metric_year('profit', y, metrics, year_labels)
        r = value_for_metric_year('revenue', y, metrics, year_labels)
        if p is not None and r is not None and r != 0:
            m = p / r
            return f"Profit margin for {y}: {m:.3f} ({m*100:.1f}%) — Profit / Revenue ({p} / {r})."
    for tok in ['revenue', 'profit', 'expenses', 'sales', 'net income']:
        if tok in q:
            desired_metric = 'revenue' if tok == 'sales' else ('profit' if tok == 'net income' else tok)
            if yrs:
                v = value_for_metric_year(desired_metric, yrs[0], summary.get("metrics", {}), year_labels)
                if v is not None:
                    return f"{desired_metric.title()} in {yrs[0]}: {v}"
    m_change = re.search(r'change.*from\s*(19\d{2}|20\d{2})\s*(?:to|-)\s*(19\d{2}|20\d{2})', q)
    if m_change:
        y1, y2 = m_change.group(1), m_change.group(2)
        met = None
        for tok in ['revenue', 'profit', 'expenses', 'sales']:
            if tok in q:
                met = 'revenue' if tok == 'sales' else tok
                break
        if met:
            v1 = value_for_metric_year(met, y1, summary.get("metrics", {}), year_labels)
            v2 = value_for_metric_year(met, y2, summary.get("metrics", {}), year_labels)
            if v1 is not None and v2 is not None:
                abs_change = v2 - v1
                pct = (abs_change / v1) * 100 if v1 != 0 else None
                if pct is None:
                    return f"{met.title()} changed by {abs_change} (previous value zero => % undefined)."
                return f"{met.title()} change from {y1} to {y2}: {abs_change} ({pct:.1f}%)."
    return None

# --------------------------
# UI: Header and Tabs
# --------------------------
st.header("💬 Financial Document Q&A — Polished")

tab_extract, tab_chat = st.tabs(["Upload & Extract", "Chat & Q&A"])

# --------------------------
# Tab: Upload & Extract
# --------------------------
with tab_extract:
    st.subheader("Upload & Extract")
    st.write("Upload an Excel (.xlsx) or PDF (.pdf) financial file. The app will extract Revenue, Expenses, Profit and save a `_summary.json` in the `uploads/` folder.")
    uploaded = st.file_uploader("Choose file", type=["pdf", "xls", "xlsx"])
    if uploaded is not None:
        saved_path = save_uploaded_file(uploaded)
        st.success(f"Saved file to `{saved_path}`")
        if uploaded.name.lower().endswith((".xls", ".xlsx")):
            summary = extract_excel_metrics(saved_path)
        else:
            summary = extract_pdf_metrics(saved_path)
        prefix = os.path.splitext(uploaded.name)[0]
        summary_path = save_summary_json(summary, prefix)
        st.success(f"Saved summary JSON to `{summary_path}`")
        st.subheader("Extracted Summary")
        st.json(summary)
        if summary.get("metrics"):
            st.subheader("Detected metrics (preview)")
            for k, v in summary["metrics"].items():
                st.write(f"**{k.title()}**")
                try:
                    vals = list(v.values())
                    formatted_vals = [f"{float(x):,.0f}" for x in vals]
                    year_labels = detect_year_labels_from_source(summary)
                    if year_labels and len(year_labels) >= len(vals):
                        df_out = pd.DataFrame({"Year": year_labels[:len(vals)], "Value": formatted_vals})
                        st.dataframe(df_out, use_container_width=True)
                    else:
                        df_out = pd.DataFrame({"Row": list(range(len(vals))), "Value": formatted_vals})
                        st.dataframe(df_out, use_container_width=True)
                except Exception:
                    st.write(v)
    else:
        st.info("Upload a file to extract metrics. If PDF parsing is desired, install `pdfplumber`.")

# --------------------------
# Tab: Chat & Q&A
# --------------------------
with tab_chat:
    st.subheader("Chat & Q&A")
    summary = load_latest_summary()
    if not summary:
        st.info("No summary found. Use Upload & Extract tab to create one.")
    else:
        st.write(f"Source: {summary.get('source')}, Type: {summary.get('type')}")
        YEAR_LABELS = detect_year_labels_from_source(summary)
        if YEAR_LABELS:
            st.write("Detected year labels:", YEAR_LABELS[:10])
        else:
            st.write("Year labels not found; UI will use numeric indices.")

        metrics = summary.get("metrics", {})
        if metrics:
            st.subheader("Metrics preview")
            col1, col2, col3 = st.columns(3)
            with col1:
                if "revenue" in metrics:
                    vals = get_metric_series(metrics["revenue"])
                    labels = YEAR_LABELS[:len(vals)] if YEAR_LABELS and len(YEAR_LABELS) >= len(vals) else [str(i) for i in range(len(vals))]
                    ser = pd.Series(vals, index=labels)
                    st.write("Revenue")
                    st.line_chart(ser, height=150)
            with col2:
                if "expenses" in metrics:
                    vals = get_metric_series(metrics["expenses"])
                    labels = YEAR_LABELS[:len(vals)] if YEAR_LABELS and len(YEAR_LABELS) >= len(vals) else [str(i) for i in range(len(vals))]
                    ser = pd.Series(vals, index=labels)
                    st.write("Expenses")
                    st.line_chart(ser, height=150)
            with col3:
                if "profit" in metrics:
                    vals = get_metric_series(metrics["profit"])
                    labels = YEAR_LABELS[:len(vals)] if YEAR_LABELS and len(YEAR_LABELS) >= len(vals) else [str(i) for i in range(len(vals))]
                    ser = pd.Series(vals, index=labels)
                    st.write("Profit")
                    st.line_chart(ser, height=150)
        else:
            st.info("No metrics found in the summary JSON.")

        # Chat area: input on one line, send below using a form (clear_on_submit=True)
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Ask a question (e.g., 'What was revenue in 2022?')", key="chat_input_widget")
            submit = st.form_submit_button("Send")
            # When form is submitted, handle below (after form block)

        # Handle submission: store last submitted message in a safe session key
        if submit and user_input and user_input.strip():
            # store last submitted message for later use
            st.session_state["chat_last_submitted"] = user_input.strip()
            st.session_state["chat_history"].append(("user", user_input.strip()))
            det = deterministic_answer(user_input, summary, YEAR_LABELS)
            if det:
                st.session_state["chat_history"].append(("assistant", det))
            else:
                if OLLAMA_AVAILABLE:
                    prompt = f"Use the following extracted metrics JSON to answer the user question concisely.\n\nMetrics:\n{json.dumps(summary.get('metrics', {}), indent=2)}\n\nQuestion: {user_input}"
                    try:
                        resp = ollama_chat(model=DEFAULT_MODEL, messages=[{"role":"user","content":prompt}])
                        out = resp.get("message", {}).get("content") if isinstance(resp, dict) else str(resp)
                        st.session_state["chat_history"].append(("assistant", out))
                    except Exception as e:
                        st.session_state["chat_history"].append(("assistant", f"Ollama call failed: {e}"))
                else:
                    st.session_state["chat_history"].append(("assistant", "No deterministic answer and Ollama not available."))

        # Chat controls
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            if st.button("Clear chat"):
                st.session_state["chat_history"] = []
                if "chat_last_submitted" in st.session_state:
                    del st.session_state["chat_last_submitted"]
        with c2:
            if st.button("Export chat"):
                out = {"chat": st.session_state.get("chat_history", [])}
                st.download_button("Download JSON", data=json.dumps(out, indent=2), file_name="chat_export.json")
        with c3:
            st.write("")

        # Render conversation
        st.subheader("Conversation")
        for role, text in st.session_state.get("chat_history", []):
            if role == "user":
                st.markdown(f"**You:** {text}")
            else:
                st.markdown(f"**Assistant:** {text}")

        # Trend plotting if the last submitted message asked for 'trend'
        last_msg = st.session_state.get("chat_last_submitted", "")
        if last_msg and 'trend' in last_msg.lower():
            metric_to_plot = None
            for t in ['revenue', 'profit', 'expenses', 'sales']:
                if t in last_msg.lower():
                    metric_to_plot = 'revenue' if t == 'sales' else t
                    break
            if metric_to_plot and metric_to_plot in metrics:
                vals = get_metric_series(metrics[metric_to_plot])
                labels = YEAR_LABELS[:len(vals)] if YEAR_LABELS and len(YEAR_LABELS) >= len(vals) else [str(i) for i in range(len(vals))]
                ser = pd.Series(vals, index=labels)
                st.subheader(f"{metric_to_plot.title()} trend")
                st.line_chart(ser)

