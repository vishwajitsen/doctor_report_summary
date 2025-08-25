"""
doctor_report_summary.py

Streamlit app for uploading medical reports (PDF / DOCX / JPG / PNG), performing advanced OCR,
and generating a patient-friendly medical summary (English) + optional Hindi translation.

Features:
- Advanced OCR: EasyOCR (preferred) with preprocessing; fallback to pytesseract.
- PDF rendering using PyMuPDF (fitz) where available.
- Local Mistral GGUF model via llama-cpp-python for medical summarization.
- Optional English -> Hindi translation via transformers (Helsinki-NLP/opus-mt-en-hi).
- Intelligent condensing: picks numeric lines and clinical keywords so prompts don't exceed context.
- Robust fallbacks to produce an informative narrative even without models.

Required (recommended) packages:
pip install streamlit pymupdf python-docx pillow pytesseract easyocr transformers torch sentencepiece llama-cpp-python opencv-python

System-level requirements:
- Tesseract OCR (for pytesseract): install via apt / brew / choco
- poppler (if you prefer pdf2image) — not required here because we use PyMuPDF

Before running:
- Place your Mistral gguf file in the same folder or set environment variable:
  LOCAL_MISTRAL_GGUF_PATH=/path/to/mistral-7b-instruct-v0.1.Q4_K_M.gguf

Run:
streamlit run doctor_report_summary.py
"""

import os
import re
import io
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import streamlit as st

# Soft imports
FITZ_AVAILABLE = False
PYPDF_AVAILABLE = False
DOCX_AVAILABLE = False
PIL_AVAILABLE = False
TESSERACT_AVAILABLE = False
EASYOCR_AVAILABLE = False
CV2_AVAILABLE = False
LLAMA_AVAILABLE = False
TRANS_AVAILABLE = False

# try imports (fail gracefully)
try:
    import fitz  # PyMuPDF for rendering PDFs
    FITZ_AVAILABLE = True
except Exception:
    pass

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except Exception:
    pass

try:
    import docx
    DOCX_AVAILABLE = True
except Exception:
    pass

try:
    from PIL import Image, ImageOps, ImageFilter
    PIL_AVAILABLE = True
except Exception:
    pass

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    pass

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    pass

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    pass

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except Exception:
    LLAMA_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANS_AVAILABLE = True
except Exception:
    TRANS_AVAILABLE = False

# ----------------- Configuration -----------------
# Detect model path candidates; allow env override
DEFAULT_GGUF_CANDIDATES = [
    "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    "/mnt/c/Users/win11/OneDrive/Documents/Report_Findings/doctor_report_summary/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    r"C:\Users\win11\OneDrive\Documents\Report_Findings\doctor_report_summary\mistral-7b-instruct-v0.1.Q4_K_M.gguf",
]
MODEL_PATH = os.getenv("LOCAL_MISTRAL_GGUF_PATH", "")
if not MODEL_PATH:
    for cand in DEFAULT_GGUF_CANDIDATES:
        if Path(cand).exists():
            MODEL_PATH = cand
            break

LLM_N_CTX = int(os.getenv("LLM_N_CTX", "4096"))
LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", "6"))
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "700"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))

MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "9000"))
HINDI_CHUNK_CHARS = int(os.getenv("HINDI_CHUNK_CHARS", "900"))

# ----------------- Utilities: OCR preprocessing -----------------
def pil_to_cv2(img: "PIL.Image.Image"):
    """Convert PIL image to OpenCV (BGR)"""
    import numpy as np
    arr = np.array(img)
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_to_pil(img):
    import numpy as np
    if img.ndim == 2:
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def preprocess_image_for_ocr(pil_img: "PIL.Image.Image", enlarge: int = 2):
    """
    Preprocess image to improve OCR: convert to grayscale, enhance contrast,
    denoise, optional binarization. Returns PIL.Image.
    """
    img = pil_img.convert("RGB")
    # upscale
    w, h = img.size
    img = img.resize((w * enlarge, h * enlarge), Image.LANCZOS)
    # convert to grayscale
    gray = ImageOps.grayscale(img)
    # use PIL filter for slight sharpening
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    if CV2_AVAILABLE:
        # further OpenCV processing
        img_cv = pil_to_cv2(gray)
        # gaussian blur
        img_cv = cv2.GaussianBlur(img_cv, (3, 3), 0)
        # adaptive threshold
        try:
            th = cv2.adaptiveThreshold(img_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 8)
            return cv2_to_pil(th)
        except Exception:
            return Image.fromarray(img_cv)
    else:
        # fallback: simple point transform
        return gray.point(lambda p: 0 if p < 200 else 255)

# ----------------- OCR functions -----------------
def ocr_with_easyocr_from_pil(images: List["PIL.Image.Image"], langs: List[str] = ["en"]) -> str:
    reader = easyocr.Reader(langs, gpu=False)  # set gpu=True if available and desired
    parts = []
    for im in images:
        # convert PIL to array
        import numpy as np
        im_arr = np.array(im)
        res = reader.readtext(im_arr, detail=0)
        if res:
            parts.append("\n".join(res))
    return "\n\n".join(parts)

def ocr_with_tesseract_from_pil(images: List["PIL.Image.Image"], lang: str = "eng") -> str:
    txt_parts = []
    for im in images:
        # pytesseract accepts PIL.Image
        try:
            txt = pytesseract.image_to_string(im, lang=lang)
        except Exception:
            txt = pytesseract.image_to_string(im)
        txt_parts.append(txt)
    return "\n\n".join(txt_parts)

def render_pdf_pages_to_pil(pdf_path: str, dpi: int = 200) -> List["PIL.Image.Image"]:
    """
    Use PyMuPDF (fitz) if available to render PDF pages to PIL images.
    """
    images = []
    if FITZ_AVAILABLE:
        doc = fitz.open(pdf_path)
        zoom = dpi / 72  # 72 dpi baseline
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            images.append(im)
        return images
    elif PYPDF_AVAILABLE:
        # fallback: extract text only
        return []
    else:
        return []

# ----------------- Text extraction for DOCX/Plain text -----------------
def extract_text_from_docx(path: str) -> str:
    if not DOCX_AVAILABLE:
        raise RuntimeError("python-docx not installed.")
    d = docx.Document(path)
    paras = [p.text for p in d.paragraphs if p.text and p.text.strip()]
    return "\n".join(paras)

# ----------------- Smart text condensing -----------------
# Regex to capture numeric lab lines (name + numeric + optional unit)
NUMERIC_LINE_RE = re.compile(
    r"(?P<name>[A-Za-z\-/\.\(\)\s]{2,80}?)\s*[:\-\t]*\s*(?P<val>[-+]?\d{1,4}(?:\.\d+)?(?:e[-+]?\d+)?)\s*(?P<unit>[a-zA-Z%/\^0-9\.\u00B5µ-]{0,12})",
    re.IGNORECASE,
)

IMAGING_KEYWORDS = [
    "ct", "mri", "x-ray", "xray", "ecg", "ekg", "echocardiography", "ultrasound", "usg",
    "pneumonia", "consolidation", "infarct", "ischemia", "lesion", "nodule", "mass",
    "degeneration", "hernia", "prolapse", "disc", "fracture", "sclerosis"
]

ECG_KEYWORDS = ["st elevation", "st depression", "t wave inversion", "qrs", "arrhythmia", "bradycardia", "tachycardia", "heart block"]

def extract_key_lines(text: str, max_lines: int = 120) -> Tuple[List[str], List[Tuple[str, float, str]]]:
    """
    Return two items:
     - condensed lines (strings) prioritized for LLM prompt
     - parsed numeric labs: list of tuples (name, value, unit)
    """

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    numeric_entries = []
    key_lines = []

    # first pass: numeric matches
    for ln in lines:
        m = NUMERIC_LINE_RE.search(ln)
        if m:
            name = m.group("name").strip()
            val = m.group("val")
            unit = (m.group("unit") or "").strip()
            try:
                fval = float(val)
            except Exception:
                try:
                    fval = float(val.replace(",", ""))
                except Exception:
                    fval = None
            if fval is not None:
                numeric_entries.append((name, fval, unit))
                key_lines.append(f"{name}: {fval} {unit}".strip())

    # second pass: imaging / ECG keywords, pick lines containing them
    for ln in lines:
        low = ln.lower()
        if any(k in low for k in IMAGING_KEYWORDS + ECG_KEYWORDS):
            key_lines.append(ln)

    # third pass: include short narrative sentences (first 6)
    sentences = re.split(r"(?<=[\.\?\!])\s+", " ".join(lines))
    for s in sentences[:6]:
        if len(s.strip()) > 20:
            key_lines.append(s.strip())

    # remove duplicates while preserving order
    seen = set()
    final_lines = []
    for ln in key_lines:
        if ln not in seen:
            final_lines.append(ln)
            seen.add(ln)
        if len(final_lines) >= max_lines:
            break

    return final_lines, numeric_entries

# ----------------- Cleaning metadata & headers -----------------
HEADER_PATTERNS = [
    r"^name[:\s]", r"^age[:\s]", r"^sex[:\s]", r"^lab\s*no[:\s]", r"^ref[:\s]",
    r"^collected[:\s]", r"^processed[:\s]", r"^final[:\s]", r"^report\s*status",
    r"^page\s*\d+"
]
HEADER_RE = re.compile("|".join(HEADER_PATTERNS), re.IGNORECASE)

def clean_report_text(text: str) -> str:
    lines = text.splitlines()
    kept = []
    for ln in lines:
        s = ln.strip()
        if not s:
            kept.append("")  # keep blank lines
            continue
        if HEADER_RE.search(s):
            continue
        # remove lines that appear to be address blocks or lab names with too many uppercase words
        if len(s) > 20 and sum(1 for ch in s if ch.isupper()) > len(s) * 0.5 and len(s.split()) > 3:
            continue
        kept.append(ln)
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned).strip()
    return cleaned

# ----------------- LLM (Mistral via llama-cpp) -----------------
@st.cache_resource(show_spinner=False)
def load_local_mistral(path: str):
    if not LLAMA_AVAILABLE:
        raise RuntimeError("llama-cpp-python not installed in this environment.")
    if not Path(path).exists():
        raise FileNotFoundError(f"GGUF model not found at: {path}")
    return Llama(model_path=str(path), n_ctx=LLM_N_CTX, n_threads=LLM_N_THREADS)

def build_mistral_prompt(condensed_lines: List[str], numeric_entries: List[Tuple[str, float, str]], cleaned_text: str) -> str:
    # Compose short prompt focusing on numeric values and key narrative lines
    header = (
        "You are a senior clinical physician and experienced medical writer. "
        "Analyze the following condensed medical report content and produce a patient-friendly, accurate, and actionable summary.\n\n"
        "Output MUST include the following labeled sections EXACTLY (use these headings):\n\n"
        "Summarised Explanation:\n- One short paragraph in plain English describing the main findings and what they likely mean.\n\n"
        "Doctor-style interpretation:\n- 2-4 concise sentences expressing clinical interpretation and urgency (if any), in clinician voice.\n\n"
        "Observations / Key Findings:\n- Short bullet points summarizing the most important abnormal results (include values when abnormal).\n\n"
        "Next steps / Checklist:\n- 4 clear actionable items (tests, referrals, lifestyle), indicate urgency if applicable.\n\n"
        "Overall Weighted Health Score out of 100:\n- Provide a number and a one-line justification.\n\n"
        "End with this exact sentence:\n"
        "End of Report. Take Care of yourself and Be happy and drink plenty of water & Exercise regularly.\n\n"
        "Important: Do NOT include patient names, IDs, addresses, lab header/metadata. Use simple language understandable by a lay patient.\n\n"
    )
    lab_section = ""
    if numeric_entries:
        lab_lines = [f"- {n}: {v} {u}".strip() for (n, v, u) in numeric_entries[:40]]
        lab_section = "Parsed numeric labs (top entries):\n" + "\n".join(lab_lines) + "\n\n"

    condensed_section = ""
    if condensed_lines:
        condensed_section = "Condensed report lines (prioritized):\n" + "\n".join(f"- {ln}" for ln in condensed_lines[:80]) + "\n\n"

    # also provide a short snippet of cleaned text (first ~1200 chars) to help the model
    snippet = cleaned_text[:1500] if cleaned_text else ""

    prompt = header + lab_section + condensed_section + "Cleaned report snippet:\n" + snippet + "\n\nPlease produce the summary now."
    # enforce max length
    if len(prompt) > MAX_INPUT_CHARS:
        prompt = prompt[:MAX_INPUT_CHARS]
    return prompt

def run_mistral(llm_handle, prompt_text: str) -> str:
    resp = llm_handle(prompt_text, max_tokens=LLM_MAX_NEW_TOKENS, temperature=LLM_TEMPERATURE, top_p=LLM_TOP_P)
    # llama_cpp returns a dict with 'choices'
    text = resp.get("choices", [{}])[0].get("text", "")
    return text.strip()

# ----------------- Translator -----------------
@st.cache_resource(show_spinner=False)
def load_translator():
    if not TRANS_AVAILABLE:
        return None
    device = 0 if (os.getenv("USE_CUDA", "0") == "1") else -1
    # Use Helsinki opus-mt en->hi
    try:
        tr = pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi", device=device)
        return tr
    except Exception:
        # fallback: attempt generic text2text pipeline with model name
        try:
            tr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi", device=device)
            return tr
        except Exception:
            return None

def translate_to_hindi(translator, text: str) -> str:
    if translator is None or not text.strip():
        return "अनुवाद उपलब्ध नहीं है।"
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out_parts = []
    for p in paras:
        if not p:
            continue
        if len(p) <= HINDI_CHUNK_CHARS:
            try:
                r = translator(p, max_length=1200)
                out_parts.append(r[0]["translation_text"])
            except Exception:
                out_parts.append(p)
        else:
            # chunk into sentences
            sents = re.split(r"(?<=[.!?])\s+", p)
            buf = ""
            for s in sents:
                if len(buf) + len(s) + 1 > HINDI_CHUNK_CHARS:
                    try:
                        r = translator(buf, max_length=1200)
                        out_parts.append(r[0]["translation_text"])
                    except Exception:
                        out_parts.append(buf)
                    buf = s
                else:
                    buf = (buf + " " + s).strip()
            if buf:
                try:
                    r = translator(buf, max_length=1200)
                    out_parts.append(r[0]["translation_text"])
                except Exception:
                    out_parts.append(buf)
    return "\n\n".join(out_parts)

# ----------------- Fallback summary builder -----------------
def compute_health_score_from_parsed(parsed: List[Tuple[str, float, str]]) -> int:
    score = 90
    for name, val, unit in parsed:
        low_name = name.lower()
        try:
            v = float(val)
        except Exception:
            continue
        # heuristics (not clinical-grade): penalize for obvious abnormalities
        if "triglycer" in low_name and v >= 150:
            score -= 20
        if low_name in ("ast", "sgot", "alt", "sgpt") and v > 40:
            score -= 10
        if "hemoglobin" in low_name and v < 12:
            score -= 12
        if "ldl" in low_name and v >= 130:
            score -= 15
        if "glucose" in low_name and v >= 200:
            score -= 25
    score = max(10, min(100, score))
    return score

def fallback_narrative(parsed: List[Tuple[str, float, str]], cleaned_text: str) -> str:
    lines = []
    lines.append("Summarised Explanation:")
    if parsed:
        # list salient numbers (top 8)
        for n, v, u in parsed[:8]:
            lines.append(f"- {n}: {v} {u}".strip())
    else:
        # short summary from cleaned text first sentences
        sents = re.split(r"(?<=[.!?])\s+", cleaned_text)
        summary = " ".join(sents[:3]).strip() if sents else "No numeric tests detected; narrative report provided."
        lines.append(summary)

    lines.append("\nDoctor-style interpretation:")
    if parsed:
        flags = []
        for n, v, u in parsed:
            nl = n.lower()
            try:
                vf = float(v)
            except Exception:
                continue
            if "triglycer" in nl and vf >= 150:
                flags.append("Elevated triglycerides — risk factor for cardiovascular disease.")
            if nl in ("ast", "alt", "sgot", "sgpt") and vf > 40:
                flags.append("Mildly elevated liver enzymes (AST/ALT) — consider fatty liver, alcohol, or drugs.")
            if "hemoglobin" in nl and vf < 12:
                flags.append("Low hemoglobin (possible anemia) — follow up if symptomatic.")
        if flags:
            lines.append("- " + " ".join(flags))
        else:
            lines.append("- No major numeric abnormalities detected; correlate with symptoms and clinical history.")
    else:
        lines.append("- Please discuss the report with your clinician for tailored interpretation.")

    score = compute_health_score_from_parsed(parsed)
    lines.append("\nObservations / Key Findings:")
    if parsed:
        for n, v, u in parsed[:10]:
            lines.append(f"- {n}: {v} {u}".strip())
    else:
        lines.append("- No clear numeric findings parsed from the document.")

    lines.append("\nNext steps / Checklist:")
    lines.append("- Share this summary with your primary care doctor or relevant specialist.")
    lines.append("- Lifestyle: balanced diet, regular exercise, maintain healthy weight.")
    lines.append("- If liver enzymes abnormal: avoid alcohol, check medications, consider ultrasound and repeat tests.")
    lines.append("- If lipids abnormal: dietary changes, exercise; consider lipid-lowering therapy after clinician review.")
    lines.append("- Repeat tests in 6-12 weeks or earlier if symptoms occur; seek urgent care for severe symptoms.")

    lines.append("\nOverall Weighted Health Score out of 100:")
    lines.append(str(score))

    lines.append("\nEnd of Report. Take Care of yourself and Be happy and drink plenty of water & Exercise regularly.")
    return "\n\n".join(lines)

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Doctor Report Analyzer (Advanced OCR + Mistral)", page_icon="🩺", layout="wide")
st.title("🩺 Doctor Report Analyzer — Advanced OCR + Mistral (Patient-friendly summaries)")

# Logo (use width to avoid Streamlit incompatibility)
if Path("vishwajit.jpg").exists():
    try:
        st.image("vishwajit.jpg", width=160)
    except Exception:
        pass

st.markdown("""
Upload a medical report (PDF / DOCX / JPG / PNG). The app will:
- perform advanced OCR (EasyOCR preferred; Tesseract fallback),
- condense and prioritize relevant information,
- use a local Mistral 7B (.gguf) via llama-cpp-python to produce a patient-friendly summary in English
  (Summarised Explanation, Doctor-style interpretation, Observations, Next steps, Health score),
- optionally translate the output to Hindi using a transformers translation model.

**Note:** This is an AI-assisted summary and does NOT replace clinical evaluation.
""")

# Sidebar controls
st.sidebar.header("Options / Models")
st.sidebar.write(f"Detected: fitz={FITZ_AVAILABLE}, easyocr={EASYOCR_AVAILABLE}, pytesseract={TESSERACT_AVAILABLE}, llama_cpp={LLAMA_AVAILABLE}, transformers={TRANS_AVAILABLE}")
use_easyocr = st.sidebar.selectbox("OCR engine (preferred)", options=["Auto (easyocr->tesseract)", "EasyOCR (if installed)", "Tesseract (if installed)"], index=0)
use_mistral = st.sidebar.checkbox("Use local Mistral 7B (.gguf) for summarization", value=LLAMA_AVAILABLE and bool(MODEL_PATH))
if use_mistral and not LLAMA_AVAILABLE:
    st.sidebar.error("llama-cpp-python not available; cannot use Mistral.")
do_translate = st.sidebar.checkbox("Also translate English summary to Hindi", value=(TRANS_AVAILABLE))

if do_translate and not TRANS_AVAILABLE:
    st.sidebar.warning("transformers not available; translation will be disabled.")

# Load models (deferred, cached)
llm = None
translator = None
if use_mistral and LLAMA_AVAILABLE and MODEL_PATH and Path(MODEL_PATH).exists():
    try:
        with st.spinner("Loading local Mistral model (may take a while)..."):
            llm = load_local_mistral(MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load Mistral model: {e}")
        llm = None
        use_mistral = False

if do_translate and TRANS_AVAILABLE:
    try:
        with st.spinner("Loading translation pipeline..."):
            translator = load_translator()
    except Exception as e:
        st.warning(f"Translation pipeline failed: {e}")
        translator = None

# File uploader
uploaded = st.file_uploader("Upload medical report (pdf/docx/jpg/png)", type=["pdf", "docx", "doc", "jpg", "jpeg", "png", "tif", "tiff", "bmp"], accept_multiple_files=False)

if uploaded:
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmpf:
        tmpf.write(uploaded.getbuffer())
        tmp_path = tmpf.name

    st.info("Extracting text (OCR if necessary). Choose 'Show OCR preview' to inspect extracted content.")
    show_preview = st.checkbox("Show OCR / extracted text preview", value=True)

    extracted_text = ""
    ocr_images_used = []

    try:
        if suffix == ".pdf":
            # render pages to images at higher DPI for better OCR
            if FITZ_AVAILABLE:
                pages = render_pdf_pages_to_pil(tmp_path, dpi=220)
                if not pages and PYPDF_AVAILABLE:
                    # text-only fallback
                    reader = PdfReader(tmp_path)
                    text_pages = []
                    for p in reader.pages:
                        try:
                            text_pages.append(p.extract_text() or "")
                        except Exception:
                            text_pages.append("")
                    extracted_text = "\n\n".join(text_pages)
                else:
                    # apply OCR to pages
                    images_for_ocr = []
                    for p in pages:
                        img_proc = preprocess_image_for_ocr(p, enlarge=2)
                        images_for_ocr.append(img_proc)
                    ocr_choice = use_easyocr
                    if use_easyocr == "Auto (easyocr->tesseract)":
                        if EASYOCR_AVAILABLE:
                            extracted_text = ocr_with_easyocr_from_pil(images_for_ocr)
                        elif TESSERACT_AVAILABLE:
                            extracted_text = ocr_with_tesseract_from_pil(images_for_ocr)
                        else:
                            extracted_text = ""
                    elif use_easyocr == "EasyOCR (if installed)":
                        if EASYOCR_AVAILABLE:
                            extracted_text = ocr_with_easyocr_from_pil(images_for_ocr)
                        else:
                            st.warning("EasyOCR not installed; falling back to Tesseract if available.")
                            if TESSERACT_AVAILABLE:
                                extracted_text = ocr_with_tesseract_from_pil(images_for_ocr)
                    else:
                        # explicit Tesseract
                        if TESSERACT_AVAILABLE:
                            extracted_text = ocr_with_tesseract_from_pil(images_for_ocr)
                        else:
                            extracted_text = ""
                    ocr_images_used = images_for_ocr
            else:
                # try reading as text via pypdf
                if PYPDF_AVAILABLE:
                    reader = PdfReader(tmp_path)
                    pages = []
                    for p in reader.pages:
                        try:
                            pages.append(p.extract_text() or "")
                        except Exception:
                            pages.append("")
                    extracted_text = "\n\n".join(pages)
                else:
                    st.error("No PDF renderer available (install pymupdf or pypdf).")
                    extracted_text = ""

        elif suffix in (".docx", ".doc"):
            try:
                extracted_text = extract_text_from_docx(tmp_path)
            except Exception as e:
                st.error(f"DOCX extraction failed: {e}")
                extracted_text = ""

        elif suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            if PIL_AVAILABLE:
                im = Image.open(tmp_path).convert("RGB")
                img_proc = preprocess_image_for_ocr(im, enlarge=2)
                if use_easyocr == "Auto (easyocr->tesseract)":
                    if EASYOCR_AVAILABLE:
                        extracted_text = ocr_with_easyocr_from_pil([img_proc])
                    elif TESSERACT_AVAILABLE:
                        extracted_text = ocr_with_tesseract_from_pil([img_proc])
                    else:
                        extracted_text = ""
                elif use_easyocr == "EasyOCR (if installed)":
                    if EASYOCR_AVAILABLE:
                        extracted_text = ocr_with_easyocr_from_pil([img_proc])
                    else:
                        st.warning("EasyOCR not installed; falling back to Tesseract if available.")
                        if TESSERACT_AVAILABLE:
                            extracted_text = ocr_with_tesseract_from_pil([img_proc])
                        else:
                            extracted_text = ""
                else:
                    if TESSERACT_AVAILABLE:
                        extracted_text = ocr_with_tesseract_from_pil([img_proc])
                    else:
                        extracted_text = ""
                ocr_images_used = [img_proc]
            else:
                st.error("Pillow not installed; cannot process image.")
                extracted_text = ""
        else:
            extracted_text = ""
    except Exception as e:
        st.error(f"Extraction/OCR failed: {e}")
        extracted_text = ""

    cleaned_text = clean_report_text(extracted_text or "")
    if show_preview:
        st.subheader("Extracted / Cleaned text preview")
        if cleaned_text.strip():
            st.text_area("Extracted text (cleaned)", cleaned_text, height=320)
        else:
            st.info("No text extracted; OCR or text extraction may have failed.")

    # condensed lines and numeric parsing
    condensed_lines, parsed_numeric = extract_key_lines(cleaned_text, max_lines=200)

    # show parsed numeric
    if parsed_numeric:
        st.subheader("Parsed numeric values (preview)")
        table_preview = "\n".join(f"{n} : {v} {u}" for n, v, u in parsed_numeric[:40])
        st.code(table_preview)

    # Generate summary
    if st.button("Generate patient-friendly summary"):
        final_english = ""
        # Prefer running local Mistral if requested and available
        if use_mistral and llm is not None:
            prompt = build_mistral_prompt(condensed_lines, parsed_numeric, cleaned_text)
            st.info("Running local Mistral summarization. This may take time on CPU.")
            try:
                with st.spinner("Generating summary with Mistral..."):
                    out = run_mistral(llm, prompt)
                # Ensure final line presence
                final_line = "End of Report. Take Care of yourself and Be happy and drink plenty of water & Exercise regularly."
                if final_line not in out:
                    out = out.strip() + "\n\n" + final_line
                final_english = out.strip()
            except Exception as e:
                st.warning(f"Mistral summarization failed: {e}. Falling back to rule-based narrative.")
                final_english = fallback_narrative(parsed_numeric, cleaned_text)
        else:
            final_english = fallback_narrative(parsed_numeric, cleaned_text)

        # Present English summary
        st.subheader("🧾 Final Structured Summary (English)")
        st.markdown(final_english.replace("\n", "  \n"))
        st.download_button("⬇️ Download English summary (.txt)", final_english, file_name="doctor_summary_en.txt")

        # Translate to Hindi if requested
        if do_translate:
            if translator is None:
                st.warning("Translator pipeline not loaded; cannot translate. Ensure transformers is installed.")
                hindi_text = "अनुवाद उपलब्ध नहीं है।"
            else:
                with st.spinner("Translating to Hindi..."):
                    try:
                        hindi_text = translate_to_hindi(translator, final_english)
                    except Exception as e:
                        st.warning(f"Translation failed: {e}")
                        hindi_text = "अनुवाद उपलब्ध नहीं है।"
            st.subheader("🧾 Final Structured Summary (Hindi)")
            st.markdown(hindi_text.replace("\n", "  \n"))
            st.download_button("⬇️ Download Hindi summary (.txt)", hindi_text, file_name="doctor_summary_hi.txt")

st.markdown("---")
st.info("This is an AI-assisted summary for informational purposes only and does not replace clinical evaluation. For urgent concerns, contact a qualified healthcare professional.")
