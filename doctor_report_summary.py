"""
doctor_report_summary.py

Streamlit app that:
- Uploads PDF / DOCX / JPG / PNG medical reports
- Extracts text (PDF / DOCX / OCR for images)
- Cleans unnecessary headers/footers from reports
- Uses a local Mistral .gguf model via llama-cpp-python
- Produces ONLY:
  1) Normal Findings
  2) Abnormal Findings
  3) Doctor-style interpretation letter
  4) Bullet checklist of improvements / next steps
  5) Overall Weighted Health Score out of 100
  Ending line: "End of Report. Take Care of yourself and Be happy and drink plenty of water & Exercise regularly."
"""

import streamlit as st
from pathlib import Path
import tempfile
import re
import pymupdf
 # PyMuPDF
import docx
from PIL import Image
import pytesseract
from llama_cpp import Llama

# ------------------------
# CONFIG
# ------------------------
MODEL_PATH = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
LLM_N_THREADS = 6
LLM_N_CTX = 4096
SAFE_MARGIN = 800   # reserve tokens for output

# ------------------------
# Load Model
# ------------------------
@st.cache_resource(show_spinner=False)
def load_llm(model_path: str):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return Llama(model_path=model_path, n_ctx=LLM_N_CTX, n_threads=LLM_N_THREADS)

try:
    llm = load_llm(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ------------------------
# Text extraction
# ------------------------
def extract_text_from_pdf(path: str) -> str:
    txt = []
    doc = pymupdf.open(path)
    for page in doc:
        txt.append(page.get_text("text"))
    return "\n".join(txt)

def extract_text_from_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_text_from_image(path: str) -> str:
    img = Image.open(path)
    return pytesseract.image_to_string(img)

def extract_text_generic(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(path)
    elif ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        return extract_text_from_image(path)
    return ""

# ------------------------
# Text cleaning (remove unnecessary metadata)
# ------------------------
def clean_report_text(text: str) -> str:
    # Remove demographics, lab addresses, headers like "Report Status", "Lab No", etc.
    patterns = [
        r"Report(ed| Status).*", 
        r"Name.*", 
        r"Lab No.*", 
        r"Ref By.*", 
        r"Collected.*", 
        r"Processed.*", 
        r"Final.*", 
        r"Address.*", 
        r"Dr\..*"
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    # collapse multiple blank lines
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()

# ------------------------
# Prompts
# ------------------------
SUMMARY_PROMPT = (
    "You are an expert physician and lab diagnostics specialist. "
    "Analyze ONLY the relevant test results from the following cleaned medical report text. "
    "Do NOT include patient demographic details, IDs, hospital names, report status, or addresses. "
    "Output MUST be structured with these five sections:\n\n"
    "1) Normal Findings:\n- Short bullet points with brief descriptions.\n\n"
    "2) Abnormal Findings:\n- Bullet points with test values and why they are abnormal.\n\n"
    "3) Doctor-style interpretation letter:\n- A plain-English letter to the patient, summarizing main issues and giving reassurance.\n\n"
    "4) Bullet checklist of improvements / next steps:\n- Clear, actionable steps (tests, lifestyle, urgent checks).\n\n"
    "5) Overall Weighted Health Score out of 100:\n- A single number with one-line justification.\n\n"
    "End with this sentence exactly:\n"
    "'End of Report. Take Care of yourself and Be happy and drink plenty of water & Exercise regularly.'\n\n"
    "Report text:\n\n"
)

# ------------------------
# Generation
# ------------------------
def generate_summary(report_text: str):
    cleaned = clean_report_text(report_text)
    truncated = cleaned[:8000]  # safe truncate
    max_out = min(LLM_N_CTX - SAFE_MARGIN, 1024)

    prompt = SUMMARY_PROMPT + truncated + "\n\nSummary:"
    resp = llm(prompt, max_tokens=max_out)
    return resp["choices"][0]["text"].strip()

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Doctor Report Analyzer", page_icon="🩺", layout="wide")

# Logo
if Path("vishwajit.jpg").exists():
    st.image("vishwajit.jpg", caption="Created by Vishwajit Sen", use_container_width=True)

st.title("🩺 Doctor Report Analyzer — Clean Summaries with Health Score")

st.markdown("""
Upload a medical report (PDF / DOCX / JPG / PNG).  
The app generates a **concise summary** with ONLY:
1. Normal Findings  
2. Abnormal Findings  
3. Doctor-style interpretation letter  
4. Bullet checklist of improvements / next steps  
5. Overall Weighted Health Score (0–100)  

👉 Ends with a motivational reminder for the patient.
""")

uploaded = st.file_uploader("Upload report", type=["pdf", "docx", "doc", "jpg", "jpeg", "png"])

if uploaded:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    report_text = extract_text_generic(tmp_path)

    if st.button("Generate Summary"):
        if not report_text.strip():
            st.error("No text extracted.")
        else:
            with st.spinner("Analyzing with AI model..."):
                summary = generate_summary(report_text)

            if summary:
                st.subheader("🧾 Final Structured Summary")
                st.markdown(summary.replace("\n", "  \n"))
                st.download_button("⬇️ Download Summary", summary, "doctor_summary.txt")

st.markdown("---")
st.info("⚠️ AI-assisted summary — not a replacement for professional medical advice.")
