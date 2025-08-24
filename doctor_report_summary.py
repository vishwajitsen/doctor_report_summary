"""
doctor_report_summary.py

- Upload PDF / DOCX / JPG / PNG medical reports
- Extract text (PDF / DOCX / OCR for images)
- Clean unnecessary metadata
- Use a local Mistral .gguf model via llama-cpp-python to produce:
    1) Normal Findings
    2) Abnormal Findings (with values)
    3) Doctor-style interpretation letter
    4) Bullet checklist of improvements / next steps
    5) Overall Weighted Health Score out of 100
- If model file missing, optionally download from Hugging Face (repo + filename configurable)
"""

import os
import sys
import time
import streamlit as st
from pathlib import Path
import tempfile
import re
from typing import Optional

# PDF / DOCX / Image handling
try:
    import fitz  # PyMuPDF (module name 'fitz' if PyMuPDF installed)
    def open_pdf(path): return fitz.open(path)
except Exception:
    try:
        import pymupdf  # fallback (some environments expose pymupdf)
        def open_pdf(path): return pymupdf.open(path)
    except Exception:
        open_pdf = None

import docx
from PIL import Image
import pytesseract

# LLM & HF hub
try:
    from llama_cpp import Llama
except Exception as e:
    st.error("llama_cpp not installed or failed to import. Install llama-cpp-python in your environment.")
    raise

# Optional HF downloader
try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# ------------------------
# CONFIG (edit if needed)
# ------------------------
# Local model filename (relative or absolute). If missing, the app can try to download.
DEFAULT_MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "mistral-7b-instruct-v0.1.Q4_K_M.gguf")
DEFAULT_MODEL_REPO = os.getenv("HF_MODEL_REPO", "TheBloke/Mistral-7B-Instruct-GGUF")  # override if different
MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", DEFAULT_MODEL_FILENAME)

# HuggingFace token (optional, needed for private models or higher rate limits)
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Llama / context settings
LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", "6"))
LLM_N_CTX = int(os.getenv("LLM_N_CTX", "4096"))
SAFE_MARGIN = int(os.getenv("SAFE_MARGIN", "800"))  # reserve tokens for prompt + output
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))

# Maximum characters of report text to send (rough guard vs tokens)
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "8000"))

# ------------------------
# Helpers: download model if missing
# ------------------------
def download_model_from_hf(repo_id: str, filename: str, token: Optional[str]=None, dest_dir: str = ".") -> Optional[str]:
    """
    Try to download `filename` from HF repo `repo_id` into dest_dir.
    Returns local path or None on failure.
    """
    if not HF_AVAILABLE:
        st.warning("huggingface_hub is not installed — cannot download model automatically.")
        return None

    st.info(f"Attempting to download model file '{filename}' from HF repo '{repo_id}'... This can be large and slow.")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        # hf_hub_download gives direct file download if filename exists
        path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        st.success(f"Model downloaded to {path}")
        return path
    except Exception as e:
        st.error(f"hf_hub_download failed: {e}")
        # fallback: try snapshot_download then locate file
        try:
            st.info("Trying snapshot_download (may take time)...")
            snap_dir = snapshot_download(repo_id=repo_id, token=token, local_dir=dest_dir)
            candidate = Path(snap_dir) / filename
            if candidate.exists():
                st.success(f"Found model in snapshot at {candidate}")
                return str(candidate)
            else:
                st.error(f"Model file not found in snapshot {snap_dir}")
        except Exception as e2:
            st.error(f"snapshot_download failed: {e2}")
    return None

# ------------------------
# Load LLM (cached)
# ------------------------
@st.cache_resource(show_spinner=False)
def load_llm(model_path: str):
    model_path = str(model_path)
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    # instantiate llama-cpp
    return Llama(model_path=model_path, n_ctx=LLM_N_CTX, n_threads=LLM_N_THREADS)

# ------------------------
# Text extraction utilities
# ------------------------
def extract_text_from_pdf(path: str) -> str:
    if open_pdf is None:
        raise RuntimeError("PDF library not available (PyMuPDF).")
    doc = open_pdf(path)
    txt_parts = []
    for page in doc:
        try:
            txt = page.get_text("text")
        except Exception:
            # some bindings expose different methods
            txt = page.get_text()
        txt_parts.append(txt)
    return "\n".join(txt_parts)

def extract_text_from_docx(path: str) -> str:
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paragraphs)

def extract_text_from_image(path: str) -> str:
    img = Image.open(path)
    # you can adjust language param e.g., lang="eng+hin" if tesseract langs installed
    text = pytesseract.image_to_string(img)
    return text

def extract_text_generic(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(path)
    elif ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        return extract_text_from_image(path)
    else:
        return ""

# ------------------------
# Clean unwanted report sections
# ------------------------
def clean_report_text(text: str) -> str:
    # Remove lines likely to be metadata or headers
    patterns = [
        r"^(Report(ed| Status)?[:\s].*)$",
        r"^Name[:\s].*$",
        r"^Lab\s*No[:\s].*$",
        r"^Ref\s*By[:\s].*$",
        r"^Collected[:\s].*$",
        r"^Processed[:\s].*$",
        r"^Final[:\s].*$",
        r"^Address[:\s].*$",
        r"^Dr\.[^\n]*$",
        r"^Report\s*Status[:\s].*$",
        r"^Test\s*Report[:\s].*$",
        r"^.*NATIONAL\s+LAB.*$",
    ]
    # remove lines matching patterns
    lines = text.splitlines()
    kept = []
    for ln in lines:
        skip = False
        for pat in patterns:
            if re.match(pat, ln.strip(), flags=re.IGNORECASE):
                skip = True
                break
        if not skip:
            kept.append(ln)
    cleaned = "\n".join(kept)
    # remove repeating whitespace
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned).strip()
    return cleaned

# ------------------------
# Prompt templates
# ------------------------
SUMMARY_PROMPT = (
    "You are an expert physician and lab diagnostics specialist. "
    "Analyze ONLY the relevant test results from the following cleaned medical report text. "
    "DO NOT include patient names, IDs, dates, hospital names, addresses, or technical headers. "
    "Output MUST EXACTLY include these five sections (use the exact headings):\n\n"
    "1) Normal Findings:\n- short bullet points (one line each) describing normal results.\n\n"
    "2) Abnormal Findings:\n- bullet points with the abnormal test, value, and a one-line explanation why it's significant.\n\n"
    "3) Doctor-style interpretation letter:\n- a brief plain-English letter to the patient (2-5 short sentences) summarizing the main issues and reassurance.\n\n"
    "4) Bullet checklist of improvements / next steps:\n- actionable numbered/bulleted steps (include urgency if needed).\n\n"
    "5) Overall Weighted Health Score out of 100:\n- single number and one-line justification.\n\n"
    "End with exactly this final line:\n"
    "End of Report. Take Care of yourself and Be happy and drink plenty of water & Exercise regularly.\n\n"
    "Report text follows:\n\n"
)

# ------------------------
# Generate summary safely (handles truncation)
# ------------------------
def generate_summary_with_llm(llm, report_text: str):
    cleaned = clean_report_text(report_text)
    if not cleaned.strip():
        return "No relevant report text could be extracted."

    # truncate to safe char length (rough heuristic)
    truncated = cleaned[:MAX_INPUT_CHARS]
    # compute safe output tokens
    safe_out = min(LLM_N_CTX - SAFE_MARGIN, MAX_OUTPUT_TOKENS)
    prompt = SUMMARY_PROMPT + truncated + "\n\nSummary:"
    # call LLM
    resp = llm(prompt, max_tokens=safe_out)
    out = resp["choices"][0]["text"].strip()
    return out

# ------------------------
# UI & flow
# ------------------------
st.set_page_config(page_title="Doctor Report Analyzer", page_icon="🩺", layout="wide")
st.title("🩺 Doctor Report Analyzer — Summaries (Mistral 7B)")

# logo
if Path("vishwajit.jpg").exists():
    st.image("vishwajit.jpg", caption="Created by Vishwajit Sen", use_container_width=True)

st.markdown(
    """
Upload a lab/medical report (PDF/DOCX/JPG/PNG). The app extracts relevant text, removes headers/IDs,
and produces a concise clinician-style summary with:
1) Normal Findings
2) Abnormal Findings (with values)
3) Doctor-style letter
4) Next steps checklist
5) Overall Weighted Health Score (0-100)

If the local model is missing the app will try to download it from Hugging Face (set HF_MODEL_REPO & HF_MODEL_FILENAME env vars if needed).
"""
)

# Show model path and status
st.sidebar.header("Model & settings")
st.sidebar.write(f"Local model path: `{MODEL_PATH}`")
st.sidebar.write(f"Context window (n_ctx): {LLM_N_CTX}, threads: {LLM_N_THREADS}")

# If model missing, try download
local_model = Path(MODEL_PATH)
if not local_model.exists():
    st.warning(f"Model file not found at {MODEL_PATH}.")
    if HF_AVAILABLE:
        st.info("Hugging Face Hub client is available; you may download model from HF instead.")
        hf_repo = st.sidebar.text_input("HF model repo (repo_id)", value=DEFAULT_MODEL_REPO)
        hf_fname = st.sidebar.text_input("HF model filename", value=DEFAULT_MODEL_FILENAME)
        if st.sidebar.button("Download model from HF"):
            with st.spinner("Downloading model from Hugging Face (may take long)..."):
                try:
                    downloaded = download_model_from_hf(hf_repo, hf_fname, token=HF_TOKEN, dest_dir=".")
                    if downloaded:
                        st.success(f"Downloaded model to {downloaded}. Please restart the app to load it.")
                        # do not auto-reload; user can refresh
                except Exception as e:
                    st.error(f"Download failed: {e}")
    else:
        st.info("To enable automatic download, install huggingface_hub and set HF_MODEL_REPO / HF_MODEL_FILENAME env vars.")
    st.stop()

# Load LLM (we cached earlier if possible)
try:
    llm = load_llm(local_model)
except Exception as e:
    st.error(f"Failed to load local model: {e}")
    st.stop()

# File uploader
uploaded = st.file_uploader("Upload medical report (PDF / DOCX / PNG / JPG)", type=["pdf","docx","doc","png","jpg","jpeg","tif","tiff"])
preview_col, action_col = st.columns([3,1])
with preview_col:
    preview = st.checkbox("Show first-page preview (if PDF/image)", value=False)
with action_col:
    do_translate = st.checkbox("Also translate English summary to Hindi", value=False)

if uploaded:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    # optional preview
    if preview:
        try:
            if suffix.lower() == ".pdf":
                doc = open_pdf(tmp_path)
                page = doc.load_page(0)
                pix = page.get_pixmap(matrix=fitz.Matrix(2,2)) if 'fitz' in sys.modules else page.get_pixmap(matrix=None)
                img_bytes = pix.tobytes("png")
                st.image(img_bytes, caption="PDF — page 1 preview")
            else:
                st.image(tmp_path, caption="Uploaded file preview")
        except Exception as e:
            st.warning(f"Preview failed: {e}")

    with st.spinner("Extracting text (OCR if necessary)..."):
        try:
            report_text = extract_text_generic(tmp_path)
        except Exception as e:
            st.error(f"Failed to extract text: {e}")
            report_text = ""

    if not report_text.strip():
        st.warning("No textual content extracted. If this is an image, try a clearer scan or run OCR with languages configured.")
    else:
        if st.button("Generate Summary"):
            with st.spinner("Generating summary (this uses the local model; may take time)..."):
                try:
                    final_en = generate_summary_with_llm(llm, report_text)
                except Exception as e:
                    st.error(f"Model inference failed: {e}")
                    final_en = ""

            if final_en:
                st.subheader("🧾 Final Structured Summary (English)")
                st.markdown(final_en.replace("\n", "  \n"))
                st.download_button("⬇️ Download English summary (.txt)", final_en, file_name="doctor_summary_en.txt")

                # Optional translate via same model
                if do_translate:
                    translate_prompt = (
                        "Translate the following English medical summary to clear, natural HINDI for a patient. "
                        "Keep headings and sections the same. English text:\n\n" + final_en + "\n\nHindi summary:"
                    )
                    with st.spinner("Translating to Hindi..."):
                        try:
                            resp = llm(translate_prompt, max_tokens=min(LLM_N_CTX - SAFE_MARGIN, MAX_OUTPUT_TOKENS))
                            final_hi = resp["choices"][0]["text"].strip()
                        except Exception as e:
                            st.error(f"Translation failed: {e}")
                            final_hi = ""
                    if final_hi:
                        st.subheader("🧾 Final Structured Summary (Hindi)")
                        st.markdown(final_hi.replace("\n", "  \n"))
                        st.download_button("⬇️ Download Hindi summary (.txt)", final_hi, file_name="doctor_summary_hi.txt")

st.info("This tool provides an AI-assisted summary for convenience. It does NOT replace clinician evaluation. For urgent concerns, contact a qualified healthcare professional immediately.")

