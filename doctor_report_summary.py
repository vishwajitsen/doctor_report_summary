"""
doctor_report_summary.py

Streamlit app that:
- Uploads PDF / DOCX / JPG / PNG medical reports
- Extracts text (PDF via PyMuPDF or PyPDF fallback / DOCX / OCR for images)
- Cleans unnecessary headers/footers from reports
- Uses a local Mistral .gguf model via llama-cpp-python
- Produces ONLY:
  1) Normal Findings
  2) Abnormal Findings (with values)
  3) Doctor-style interpretation letter
  4) Bullet checklist of improvements / next steps
  5) Overall Weighted Health Score out of 100

If the model file is missing, you can optionally download it from Hugging Face.
"""

import os
import sys
import re
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

# ------------------------
# Soft imports (NO hard crash on Cloud)
# ------------------------
# PDF engines
FITZ_AVAILABLE = False
PYMUPDF_AVAILABLE = False
PYPDF_AVAILABLE = False

try:
    import fitz  # PyMuPDF canonical import
    FITZ_AVAILABLE = True
    def _open_pdf(path): return fitz.open(path)
except Exception:
    try:
        import pymupdf  # some envs expose pymupdf
        PYMUPDF_AVAILABLE = True
        def _open_pdf(path): return pymupdf.open(path)
    except Exception:
        try:
            from pypdf import PdfReader
            PYPDF_AVAILABLE = True
        except Exception:
            pass

# DOCX
DOCX_AVAILABLE = False
try:
    import docx
    DOCX_AVAILABLE = True
except Exception:
    pass

# Images & OCR
PIL_AVAILABLE = False
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    pass

TESSERACT_AVAILABLE = False
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    pass

# LLM
LLAMA_AVAILABLE = True
try:
    from llama_cpp import Llama
except Exception:
    LLAMA_AVAILABLE = False

# Optional HF downloader
HF_AVAILABLE = False
try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_AVAILABLE = True
except Exception:
    pass


# ------------------------
# CONFIG (edit if needed)
# ------------------------
DEFAULT_MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "mistral-7b-instruct-v0.1.Q4_K_M.gguf")
DEFAULT_MODEL_REPO = os.getenv("HF_MODEL_REPO", "TheBloke/Mistral-7B-Instruct-GGUF")
MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", DEFAULT_MODEL_FILENAME)

HF_TOKEN = os.getenv("HF_TOKEN", None)

LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", "6"))
LLM_N_CTX = int(os.getenv("LLM_N_CTX", "4096"))
SAFE_MARGIN = int(os.getenv("SAFE_MARGIN", "800"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "8000"))  # rough guard against long prompts


# ------------------------
# Helpers: download model if missing
# ------------------------
def download_model_from_hf(repo_id: str, filename: str, token: Optional[str] = None, dest_dir: str = ".") -> Optional[str]:
    if not HF_AVAILABLE:
        st.warning("huggingface_hub is not installed — cannot download model automatically.")
        return None
    st.info(f"Attempting to download model file '{filename}' from HF repo '{repo_id}'. This can be large.")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        st.success(f"Model downloaded to {path}")
        return path
    except Exception as e:
        st.error(f"hf_hub_download failed: {e}")
        try:
            st.info("Trying snapshot_download (may take longer)...")
            snap_dir = snapshot_download(repo_id=repo_id, token=token, local_dir=dest_dir)
            candidate = Path(snap_dir) / filename
            if candidate.exists():
                st.success(f"Found model in snapshot at {candidate}")
                return str(candidate)
            st.error(f"Model file '{filename}' not found in snapshot {snap_dir}")
        except Exception as e2:
            st.error(f"snapshot_download failed: {e2}")
    return None


# ------------------------
# Load LLM (cached)
# ------------------------
@st.cache_resource(show_spinner=False)
def load_llm(model_path: str):
    if not LLAMA_AVAILABLE:
        raise RuntimeError("llama_cpp not installed. Add 'llama-cpp-python' to requirements.txt.")
    model_path = str(model_path)
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return Llama(model_path=model_path, n_ctx=LLM_N_CTX, n_threads=LLM_N_THREADS)


# ------------------------
# Extraction helpers
# ------------------------
def extract_text_from_pdf(path: str) -> str:
    # Prefer PyMuPDF if available (best layout text)
    if FITZ_AVAILABLE or PYMUPDF_AVAILABLE:
        doc = _open_pdf(path)
        parts = []
        for page in doc:
            try:
                parts.append(page.get_text("text"))
            except Exception:
                parts.append(page.get_text())
        return "\n".join(parts)

    # Fallback: PyPDF
    if PYPDF_AVAILABLE:
        reader = PdfReader(path)
        parts = []
        for p in reader.pages:
            try:
                parts.append(p.extract_text() or "")
            except Exception:
                parts.append("")
        return "\n".join(parts)

    raise RuntimeError("No PDF extractor available. Install 'pymupdf' or 'pypdf'.")


def extract_text_from_docx(path: str) -> str:
    if not DOCX_AVAILABLE:
        raise RuntimeError("python-docx not installed. Add 'python-docx' to requirements.txt.")
    d = docx.Document(path)
    paras = [p.text for p in d.paragraphs if p.text and p.text.strip()]
    return "\n".join(paras)


def extract_text_from_image(path: str) -> str:
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow not installed. Add 'pillow' to requirements.txt.")
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("pytesseract not installed. Add 'pytesseract' to requirements.txt and ensure Tesseract is available.")
    img = Image.open(path)
    return pytesseract.image_to_string(img)


def extract_text_generic(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext in (".docx", ".doc"):
        return extract_text_from_docx(path)
    if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        return extract_text_from_image(path)
    return ""


# ------------------------
# Cleaning
# ------------------------
def clean_report_text(text: str) -> str:
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
    lines = text.splitlines()
    kept = []
    for ln in lines:
        if not any(re.match(p, ln.strip(), flags=re.IGNORECASE) for p in patterns):
            kept.append(ln)
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned).strip()
    return cleaned


# ------------------------
# Prompt
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


def generate_summary_with_llm(llm, report_text: str) -> str:
    cleaned = clean_report_text(report_text)
    if not cleaned.strip():
        return "No relevant report text could be extracted."
    truncated = cleaned[:MAX_INPUT_CHARS]
    safe_out = min(LLM_N_CTX - SAFE_MARGIN, MAX_OUTPUT_TOKENS)
    prompt = SUMMARY_PROMPT + truncated + "\n\nSummary:"
    resp = llm(prompt, max_tokens=safe_out)
    return resp["choices"][0]["text"].strip()


# ------------------------
# UI
# ------------------------
st.set_page_config(page_title="Doctor Report Analyzer", page_icon="🩺", layout="wide")
st.title("🩺 Doctor Report Analyzer — Summaries (Mistral 7B)")

if Path("vishwajit.jpg").exists():
    st.image("vishwajit.jpg", caption="Created by Vishwajit Sen", use_container_width=True)

# Capability notices (so the app doesn't just crash)
missing = []
if not (FITZ_AVAILABLE or PYMUPDF_AVAILABLE or PYPDF_AVAILABLE):
    missing.append("PDF extractor (install `pymupdf` or `pypdf`)")
if not DOCX_AVAILABLE:
    missing.append("DOCX support (`python-docx`)")
if not PIL_AVAILABLE:
    missing.append("Image support (`pillow`)")
if PIL_AVAILABLE and not TESSERACT_AVAILABLE:
    missing.append("OCR engine (`pytesseract` + system Tesseract)")
if not LLAMA_AVAILABLE:
    missing.append("LLM backend (`llama-cpp-python`)")

if missing:
    st.warning("Some features are unavailable: " + ", ".join(missing))

st.markdown("""
Upload a lab/medical report (PDF/DOCX/JPG/PNG).  
The app extracts text, removes headers/IDs, and generates a clinician-style summary:
1) Normal Findings  
2) Abnormal Findings (with values)  
3) Doctor-style letter  
4) Next steps checklist  
5) Overall Weighted Health Score (0–100)  
""")

# Sidebar
st.sidebar.header("Model & settings")
st.sidebar.write(f"Local model path: `{MODEL_PATH}`")
st.sidebar.write(f"Context window (n_ctx): {LLM_N_CTX}, threads: {LLM_N_THREADS}")

# Model presence / download
local_model = Path(MODEL_PATH)
if not local_model.exists():
    st.warning(f"Model file not found at `{MODEL_PATH}`.")
    if HF_AVAILABLE:
        repo = st.sidebar.text_input("HF model repo", value=DEFAULT_MODEL_REPO)
        fname = st.sidebar.text_input("HF model filename", value=DEFAULT_MODEL_FILENAME)
        if st.sidebar.button("Download model from HF"):
            with st.spinner("Downloading model from Hugging Face..."):
                path = download_model_from_hf(repo, fname, token=HF_TOKEN, dest_dir=".")
                if path:
                    st.success(f"Downloaded to {path}. Please rerun or refresh.")
    else:
        st.info("To enable auto-download, add `huggingface_hub` to requirements.txt.")
    st.stop()

# Load model (safe)
try:
    llm = load_llm(str(local_model))
except Exception as e:
    st.error(f"Failed to load local model: {e}")
    st.stop()

# Uploader
uploaded = st.file_uploader(
    "Upload medical report",
    type=["pdf", "docx", "doc", "png", "jpg", "jpeg", "tif", "tiff"]
)

col1, col2 = st.columns([3, 1])
with col1:
    preview = st.checkbox("Show first-page preview (PDF/image)", value=False)
with col2:
    do_translate = st.checkbox("Also translate English summary to Hindi", value=False)

if uploaded:
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    # Preview (only if we can)
    if preview:
        try:
            if suffix == ".pdf" and (FITZ_AVAILABLE or PYMUPDF_AVAILABLE):
                doc = _open_pdf(tmp_path)
                page = doc.load_page(0)
                # Only use Matrix if fitz exists
                if FITZ_AVAILABLE:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                else:
                    pix = page.get_pixmap()
                st.image(pix.tobytes("png"), caption="PDF — page 1 preview")
            elif suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff") and PIL_AVAILABLE:
                st.image(tmp_path, caption="Uploaded image preview")
        except Exception as e:
            st.warning(f"Preview failed (non-blocking): {e}")

    # Extract
    with st.spinner("Extracting text (OCR if necessary)..."):
        try:
            report_text = extract_text_generic(tmp_path)
        except Exception as e:
            st.error(f"Failed to extract text: {e}")
            report_text = ""

    if not report_text.strip():
        st.warning("No textual content extracted. If this is an image, ensure a clear scan or enable OCR dependencies.")
    else:
        if st.button("Generate Summary"):
            with st.spinner("Generating summary with the local model..."):
                try:
                    final_en = generate_summary_with_llm(llm, report_text)
                except Exception as e:
                    st.error(f"Model inference failed: {e}")
                    final_en = ""

            if final_en:
                st.subheader("🧾 Final Structured Summary (English)")
                st.markdown(final_en.replace("\n", "  \n"))
                st.download_button("⬇️ Download English summary (.txt)", final_en, file_name="doctor_summary_en.txt")

                if do_translate:
                    translate_prompt = (
                        "Translate the following English medical summary to clear, natural HINDI for a patient. "
                        "Keep headings and sections the same.\n\nEnglish text:\n\n"
                        + final_en + "\n\nHindi summary:"
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

st.markdown("---")
st.info("This tool provides an AI-assisted summary for convenience. It does NOT replace clinician evaluation.")
