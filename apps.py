# app.py — MeetEase (FAST, single-file, NO DB)
# --------------------------------------------------------------------------------------
# - No database required (stores small JSON artifacts under ./cache).
# - OCR: PyMuPDF + Tesseract (optional).
# - Indexing: FAISS (if available) + BM25 with disk caching.
# - STT: faster-whisper (CTranslate2) with ffmpeg/pydub fallback.
# - LLM: OpenAI optional (paste key in sidebar); graceful fallbacks if not set.
# --------------------------------------------------------------------------------------

from __future__ import annotations

import os, io, re, csv, json, time, math, pickle, hashlib, tempfile, warnings, gc, subprocess, shutil, sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from datetime import date, datetime

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================== CONFIG ===============================
# You can override via environment variables if you want
OPENAI_MODEL       = os.getenv("MEETEASE_OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("MEETEASE_OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Embeddings
EMBED_MODE         = os.getenv("MEETEASE_EMBED_MODE", "minilm").lower()  # 'minilm' | 'openai'
MINILM_MODEL_NAME  = os.getenv("MEETEASE_MINILM", "sentence-transformers/all-MiniLM-L6-v2")

# Whisper / OCR / Tokenization
WHISPER_MODEL      = os.getenv("MEETEASE_WHISPER_MODEL", "base")
TEMPERATURE        = float(os.getenv("MEETEASE_TEMPERATURE", "0.2"))
MAX_INPUT_TOKENS   = int(os.getenv("MEETEASE_MAX_INPUT_TOKENS", "3000"))
TESSERACT_PATH_WIN = os.getenv("MEETEASE_TESSERACT_PATH", "")

# Caches & uploads
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
CACHE_DIR  = os.getenv("CACHE_DIR", "cache")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)

# ============================== UI ===============================
import streamlit as st
st.set_page_config(page_title="MeetEase — Meeting Management (No-DB)", page_icon="🎯", layout="wide")
st.markdown("""
<style>
.main .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1200px;}
.big-title {font-size: 2rem; font-weight: 800; margin-bottom: .25rem;}
.subtle {color: #6b7280;}
.card {padding: 1rem 1.25rem; border: 1px solid #e5e7eb; border-radius: 14px; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,.04);} 
.card h4 {margin: 0 0 .5rem 0;}
.kv {display:flex; gap:.5rem; align-items:center;}
.kv b{min-width:150px; display:inline-block;}
.stButton>button {border-radius: 10px; padding: .5rem 1rem;}
textarea {border-radius: 10px !important;}
.streamlit-expanderHeader {font-weight: 700;}
.progress-wrap {border:1px solid #e5e7eb; border-radius: 10px; padding:.75rem; margin:.5rem 0;}
.small {font-size:.9rem; color:#6b7280}
.codebox {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🤖 MeetEase — Meeting Management (No-DB)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Prepare, run, and summarize meetings with AI assistance — all cached locally.</div>', unsafe_allow_html=True)
st.write("")

# Sidebar: OpenAI key (session-only, not persisted)
st.sidebar.header("🔐 API Keys")
if "OPENAI_API_KEY" not in st.session_state:
    # initialize from environment if present, else empty
    st.session_state.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
st.session_state.OPENAI_API_KEY = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    value=st.session_state.OPENAI_API_KEY or "",
    help="Used for agenda/summary generation and (optionally) OpenAI embeddings. Not saved to disk."
)
# Propagate to environment for langchain_openai
if st.session_state.OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = st.session_state.OPENAI_API_KEY

# ============================== FILE STORAGE (NO DB) ===============================
def _slug(s: str) -> str:
    s = re.sub(r"[^\w\-]+", "-", s.strip().lower()).strip("-")
    return re.sub(r"-+", "-", s) or "session"

def _session_id(title: str, mdate: date, doc_hash: str) -> str:
    base = f"{_slug(title)}-{mdate.isoformat()}-{doc_hash[:8]}"
    return base

def _sess_dir(session_id: str) -> str:
    d = os.path.join(CACHE_DIR, session_id)
    os.makedirs(d, exist_ok=True)
    return d

def _write_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")

def _read_text(path: str) -> str:
    if not os.path.isfile(path): return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _write_json(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _read_json(path: str) -> Dict:
    if not os.path.isfile(path): return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================== OCR / FILES ===============================
from PIL import Image
import numpy as np
import pytesseract
import fitz  # PyMuPDF
import docx
import cv2

if TESSERACT_PATH_WIN:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH_WIN

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def settings_hash(d: Dict) -> str:
    s = json.dumps(d, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()

# ============================== TOKENIZER ===============================
@st.cache_resource(show_spinner=False)
def token_encoder_cached():
    try:
        import tiktoken
        try: return tiktoken.encoding_for_model(OPENAI_MODEL)
        except Exception: return tiktoken.get_encoding("cl100k_base")
    except Exception:
        class _Dummy:
            def encode(self, x): return list(x.encode("utf-8"))
            def decode(self, ids): return bytes(ids).decode("utf-8", errors="ignore")
        return _Dummy()

def truncate_tokens(text: str, max_tokens: int) -> str:
    enc = token_encoder_cached()
    ids = enc.encode(text or "")
    return enc.decode(ids[:max_tokens])

# ============================== EMBEDDINGS / VECTORS ===============================
FAISS_AVAILABLE = True
try:
    from langchain_community.vectorstores import FAISS
except Exception:
    FAISS_AVAILABLE = False
    FAISS = None

HF_OK = True
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception:
    HF_OK = False
    HuggingFaceEmbeddings = None

OPENAI_EMB_OK = True
try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    OPENAI_EMB_OK = False
    OpenAIEmbeddings = None

from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter

@st.cache_resource(show_spinner=False)
def get_embeddings_cached():
    # Use OpenAI embeddings only if a key is present and the package is available
    if EMBED_MODE == "openai" and st.session_state.OPENAI_API_KEY and OPENAI_EMB_OK:
        return OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    if HF_OK:
        return HuggingFaceEmbeddings(model_name=MINILM_MODEL_NAME, encode_kwargs={"normalize_embeddings": True})
    # Minimal dummy fallback
    class _Tiny:
        def embed_documents(self, texts): return [[hashlib.md5(t.encode()).hexdigest().__hash__()%997] for t in texts]
        def embed_query(self, text): return [hashlib.md5(text.encode()).hexdigest().__hash__()%997]
    return _Tiny()

@st.cache_resource(show_spinner=False)
def get_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=160)

def build_chunks(text: str) -> List[str]:
    splitter = get_splitter()
    chunks = splitter.split_text(text or "")
    return chunks if chunks else [text or ""]

def faiss_save(store, path_dir: str):
    if FAISS_AVAILABLE and store is not None:
        os.makedirs(path_dir, exist_ok=True)
        store.save_local(path_dir)

def faiss_load(path_dir: str, embeddings):
    if not FAISS_AVAILABLE or not os.path.isdir(path_dir): 
        return None
    try:
        return FAISS.load_local(path_dir, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return None

def bm25_save(bm25: BM25Okapi, path_file: str):
    os.makedirs(os.path.dirname(path_file), exist_ok=True)
    with open(path_file, "wb") as f:
        pickle.dump(bm25, f)

def bm25_load(path_file: str) -> Optional[BM25Okapi]:
    if not os.path.isfile(path_file): return None
    with open(path_file, "rb") as f:
        return pickle.load(f)

# ============================== LLM ===============================
CHAT_OK = True
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
except Exception:
    CHAT_OK = False
    ChatOpenAI = PromptTemplate = LLMChain = None

if CHAT_OK:
    AGENDA_PROMPT = PromptTemplate(
        input_variables=["discussion_points", "context"],
        template=(
            "You are a project coordinator. Create a concise, well-structured meeting agenda.\n\n"
            "Discussion points:\n{discussion_points}\n\n"
            "Relevant context (from docs):\n{context}\n\n"
            "Return a professional agenda with clear sections and a logical flow."
        ),
    )
    SUMMARY_PROMPT_JSON = PromptTemplate(
        input_variables=["ctx", "transcript", "query"],
        template=(
            "Using the context and transcript, produce a crisp post-meeting summary with:\n"
            "1) Key Discussion Topics\n2) Decisions Made\n3) Action Items (owner & due date if stated)\n\n"
            "Context:\n{ctx}\n\nTranscript:\n{transcript}\n\nQuery:\n{query}\n"
            "Return JSON with keys: topics, decisions, action_items."
        ),
    )

@st.cache_resource(show_spinner=False)
def maybe_llm(max_tokens=400, temperature=0.2):
    if not (st.session_state.OPENAI_API_KEY and CHAT_OK):
        return None
    return ChatOpenAI(model=OPENAI_MODEL, temperature=temperature, max_tokens=max_tokens)

def run_json(chain, **kwargs) -> Dict:
    if chain is None:
        return {"topics": [], "decisions": [], "action_items": []}
    out = chain.run(**kwargs).strip()
    try:
        obj = json.loads(out)
        if isinstance(obj, dict): return obj
    except Exception:
        pass
    m = re.search(r"{.*}", re.sub(r"```json|```", "", out), flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"topics": [], "decisions": [], "action_items": []}

def dedupe_lines(text: str) -> str:
    seen, out = set(), []
    for line in [l.strip() for l in (text or "").splitlines() if l.strip()]:
        key = re.sub(r"\W+","", line.lower())
        if key in seen: continue
        seen.add(key); out.append(line)
    return "\n".join(out)

# ============================== STT ===============================
FW_OK = True
try:
    from faster_whisper import WhisperModel
except Exception:
    FW_OK = False
    WhisperModel = None

from pydub import AudioSegment

@st.cache_resource(show_spinner=False)
def load_whisper(model_name: str):
    if not FW_OK:
        return None
    device = "cuda" if (os.getenv("CUDA_VISIBLE_DEVICES") not in (None, "", "-1")) else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(model_name, device=device, compute_type=compute_type)

def safe_tmp_path(suffix=".wav") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path

def safe_unlink(path: str, max_tries: int = 6, wait_s: float = 0.25):
    for i in range(max_tries):
        try:
            if os.path.exists(path):
                os.remove(path)
            return
        except PermissionError:
            time.sleep(wait_s * (i + 1))
            gc.collect()
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

def extract_audio_to_wav(media_path: str) -> str:
    out_wav = safe_tmp_path(".wav")
    ff = shutil.which("ffmpeg")
    if ff:
        subprocess.run([ff, "-y", "-i", media_path, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", out_wav],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    else:
        # Fallback: pydub (works if codecs present in the environment)
        AudioSegment.from_file(media_path).set_frame_rate(16000).set_channels(1).export(out_wav, format="wav")
    return out_wav

def transcribe_long_audio(audio_path: str, progress_cb=None) -> str:
    model = load_whisper(WHISPER_MODEL)
    if model is None:
        return "(Transcription unavailable: faster-whisper not installed on host)"
    segments, info = model.transcribe(
        audio_path, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=800)
    )
    text_parts, total_dur = [], (getattr(info, "duration", None) or 1.0)
    last_t = 0.0
    for seg in segments:
        text_parts.append(seg.text.strip())
        last_t = getattr(seg, "end", None) or last_t
        if progress_cb:
            p = min(1.0, last_t / total_dur)
            progress_cb(p, max(0.0, total_dur - last_t))
    final = " ".join(t for t in text_parts if t)
    final = re.sub(r"\s+", " ", final).strip()
    return dedupe_lines(final)

# ============================== OCR EXTRACTORS ===============================
def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(th > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45: angle = 90 + angle
        M = cv2.getRotationMatrix2D((th.shape[1]//2, th.shape[0]//2), angle, 1.0)
        th = cv2.warpAffine(th, M, (th.shape[1], th.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    h, w = th.shape[:2]
    if max(h, w) < 1000:
        scale = 1000 / max(h, w)
        th = cv2.resize(th, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return th

def pil_ocr(img_bgr: np.ndarray) -> str:
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return pytesseract.image_to_string(pil_img)

@st.cache_data(show_spinner=False)
def cached_extract_pdf_text(doc_hash: str, pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = []
    for i in range(doc.page_count):
        pg = doc.load_page(i)
        raw = pg.get_text("text").strip()
        if len(raw) < 80:
            pix = pg.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            proc = preprocess_for_ocr(bgr)
            raw = pytesseract.image_to_string(Image.fromarray(proc))
        out.append(raw)
    return "\n".join(out)

@st.cache_data(show_spinner=False)
def cached_extract_docx_text(doc_hash: str, doc_bytes: bytes) -> str:
    d = docx.Document(io.BytesIO(doc_bytes))
    return "\n".join(p.text for p in d.paragraphs)

@st.cache_data(show_spinner=False)
def cached_extract_image_text(doc_hash: str, img_bytes: bytes) -> str:
    arr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    raw = pil_ocr(bgr)
    if len((raw or "").strip()) < 20:
        proc = preprocess_for_ocr(bgr)
        raw = pytesseract.image_to_string(Image.fromarray(proc))
    return raw

# ============================== RAG HELPERS ===============================
def build_and_persist_indices(session_id: str, doc_hash: str, full_text: str, embed_mode: str, splitter_conf: Dict):
    s_hash = settings_hash({
        "embed_mode": embed_mode,
        "model": MINILM_MODEL_NAME if embed_mode=="minilm" else OPENAI_EMBED_MODEL,
        "splitter": splitter_conf,
        "doc_hash": doc_hash,
    })
    sess_dir = _sess_dir(session_id)
    faiss_dir = os.path.join(sess_dir, f"faiss_{s_hash[:8]}")
    bm25_path = os.path.join(sess_dir, f"bm25_{s_hash[:8]}.pkl")

    embeddings = get_embeddings_cached()
    chunks = build_chunks(full_text)

    store = None
    if FAISS_AVAILABLE:
        if os.path.isdir(faiss_dir):
            store = faiss_load(faiss_dir, embeddings)
        if store is None:
            try:
                from langchain_community.vectorstores import FAISS as _FAISS
                store = _FAISS.from_texts(chunks, embeddings)
                faiss_save(store, faiss_dir)
            except Exception:
                store = None  # continue with BM25

    bm25 = bm25_load(bm25_path)
    if bm25 is None:
        bm25 = BM25Okapi([c.split() for c in chunks])
        bm25_save(bm25, bm25_path)

    # persist chunks too (for debug/inspection)
    _write_json(os.path.join(sess_dir, "chunks.json"), {"chunks": chunks})
    # remember index metadata
    _write_json(os.path.join(sess_dir, "indices.json"), {
        "doc_hash": doc_hash, "bm25_path": bm25_path, "faiss_path": faiss_dir if store else "", 
        "embed_mode": ("openai:"+OPENAI_EMBED_MODEL) if EMBED_MODE=="openai" else ("hf:"+MINILM_MODEL_NAME)
    })
    return store, bm25, chunks, faiss_dir, bm25_path

def select_context(store, bm25, chunks: List[str], query: str, k:int=4) -> str:
    ctx = ""
    if store:
        try:
            docs = store.similarity_search(query, k=k)
            ctx = "\n\n".join(d.page_content for d in docs)
        except Exception:
            pass
    if (not ctx) and bm25 and chunks:
        import numpy as _np
        scores = bm25.get_scores(query.split())
        top_idx = _np.argsort(scores)[-k:][::-1]
        ctx = "\n\n".join(chunks[i] for i in top_idx)
    return ctx

def analyze_agenda_resolution(agenda_points: List[str], transcript: str) -> Tuple[List[str], List[str]]:
    t = (transcript or "").lower()
    resolved_kw = ["resolved", "completed", "closed", "fixed", "agreed"]
    unresolved_kw = ["unresolved", "pending", "needs further discussion", "open", "incomplete"]
    resolved, unresolved = [], []
    for p in agenda_points:
        pl = p.lower()
        if pl in t:
            idx = t.find(pl)
            window = t[max(0, idx-200): idx+200]
            if any(k in window for k in resolved_kw): resolved.append(p)
            elif any(k in window for k in unresolved_kw): unresolved.append(p)
            else: unresolved.append(p)
        else:
            unresolved.append(p)
    return resolved, unresolved

# ============================== DIAGNOSTICS ===============================
with st.expander("🔧 Diagnostics (click to expand)"):
    info_cols = st.columns(2)
    with info_cols[0]:
        st.write("**Python**", sys.version.split()[0])
        st.write("**Platform**", sys.platform)
        st.write("**ffmpeg**", shutil.which("ffmpeg") or "not-found")
        st.write("**tesseract**", shutil.which("tesseract") or (TESSERACT_PATH_WIN or "not-found"))
        st.write("**faster-whisper**", "OK" if FW_OK else "missing")
        st.write("**FAISS**", "OK" if FAISS_AVAILABLE else "missing (BM25 fallback)")
        st.write("**HF Embeddings**", "OK" if HF_OK else "missing")
        st.write("**OpenAI (LLM/Emb)**", "enabled" if st.session_state.OPENAI_API_KEY else "disabled")
    with info_cols[1]:
        sess_count = len([d for d in os.listdir(CACHE_DIR) if os.path.isdir(os.path.join(CACHE_DIR, d))])
        st.write("**Cache dir**", os.path.abspath(CACHE_DIR))
        st.write("**Sessions cached**", sess_count)

# ============================== APP STATE ===============================
@dataclass
class AppState:
    session_id: Optional[str] = None
    document_hash: Optional[str] = None
    discussion_points: Optional[List[str]] = None
    chunks: Optional[List[str]] = None
    faiss_store: Optional[object] = None
    bm25: Optional[BM25Okapi] = None
    last_transcript: Optional[str] = None
    last_doc_text: Optional[str] = None
    last_agenda: Optional[str] = None
    last_summary: Optional[Dict] = None

if "app" not in st.session_state:
    st.session_state.app = AppState()
app: AppState = st.session_state.app

# ============================== TABS ===============================
tab_pre, tab_agenda, tab_track, tab_summary = st.tabs(
    ["📄 Pre-Meeting", "📋 Agenda", "🎥 Tracking", "📝 Post-Summary"]
)

# ---------------- PRE-MEETING ----------------
with tab_pre:
    st.markdown("### 📄 Pre-Meeting — Documents & Discussion Points")
    colA, colB = st.columns([1.15, 1])

    with colA:
        title = st.text_input("Meeting title", value="Weekly Sync")
        mdate = st.date_input("Meeting date", value=date.today())
        up = st.file_uploader("Upload a document (PDF, DOCX, or Image)", type=["pdf", "docx", "png", "jpg", "jpeg"])
        dpoints = st.text_area("Discussion points (comma-separated)", placeholder="Budget approval, Roadmap alignment, Risk review ...", height=120)

        ocr_bar = st.progress(0.0, text="Waiting for document...")

        if st.button("Process & Build Context", type="primary", use_container_width=True):
            if not title.strip():
                st.warning("Please enter a meeting title."); st.stop()
            if not up or not dpoints.strip():
                st.warning("Please upload a document and add discussion points."); st.stop()

            raw = up.read()
            doc_hash = sha256_bytes(raw)
            app.document_hash = doc_hash
            session_id = _session_id(title.strip(), mdate, doc_hash)
            app.session_id = session_id
            sess_dir = _sess_dir(session_id)

            # Extract or reuse text
            extracted_path = os.path.join(sess_dir, "extracted.txt")
            if os.path.isfile(extracted_path):
                text = _read_text(extracted_path)
                ocr_bar.progress(1.0, text="Reusing cached text")
            else:
                ext = (up.name.split(".")[-1] or "").lower()
                if ext == "pdf":
                    ocr_bar.progress(0.2, text="Extracting PDF text...")
                    text = cached_extract_pdf_text(doc_hash, raw)
                    ocr_bar.progress(1.0, text="PDF processed")
                elif ext == "docx":
                    ocr_bar.progress(0.3, text="Extracting from DOCX...")
                    text = cached_extract_docx_text(doc_hash, raw)
                    ocr_bar.progress(1.0, text="DOCX extracted")
                elif ext in ("png","jpg","jpeg"):
                    ocr_bar.progress(0.3, text="OCR image...")
                    text = cached_extract_image_text(doc_hash, raw)
                    ocr_bar.progress(1.0, text="Image OCR done")
                else:
                    st.error("Unsupported file type."); st.stop()
                _write_text(extracted_path, text)

            app.last_doc_text = text

            # Build or load indices
            st.info("Building / loading indices...")
            splitter_conf = {"chunk_size":1400, "overlap":160}
            app.faiss_store, app.bm25, app.chunks, _, _ = build_and_persist_indices(
                session_id, doc_hash, app.last_doc_text or "", EMBED_MODE, splitter_conf
            )

            app.discussion_points = [p.strip() for p in dpoints.split(",") if p.strip()]
            _write_json(os.path.join(sess_dir, "meta.json"), {
                "title": title.strip(), "date": mdate.isoformat(), "discussion_points": app.discussion_points
            })

            st.success(f"Context ready! Session: `{session_id}` → Go to **Agenda** to generate an agenda.")

    with colB:
        st.markdown("#### Document Snapshot")
        if app.last_doc_text:
            preview = app.last_doc_text or ""
            token_count = len(preview.split())
            st.markdown(f'<div class="card"><div class="kv"><b>Words</b> {token_count}</div><div class="kv"><b>Preview</b></div></div>', unsafe_allow_html=True)
            with st.expander("Show extracted text (first 1200 chars)"):
                st.write(preview[:1200] + ("..." if len(preview) > 1200 else ""))
        else:
            st.info("No document processed yet.")

# ---------------- AGENDA ----------------
with tab_agenda:
    st.markdown("### 📋 Agenda Creation")
    if not (app.session_id and app.discussion_points and (app.faiss_store or app.bm25)):
        st.warning("Please complete **Pre-Meeting** first.")
    else:
        st.markdown("**Your discussion points**")
        st.write(", ".join(app.discussion_points))

        if st.button("Generate Agenda", type="primary", use_container_width=True):
            q = " ".join(app.discussion_points)
            ctx = select_context(app.faiss_store, app.bm25, app.chunks or [], q, k=4)
            llm = maybe_llm(max_tokens=400, temperature=TEMPERATURE)
            if CHAT_OK and llm:
                chain = LLMChain(prompt=AGENDA_PROMPT, llm=llm)
                in_ctx = truncate_tokens(ctx, MAX_INPUT_TOKENS // 2)
                agenda_md = chain.run(discussion_points="\n- " + "\n- ".join(app.discussion_points), context=in_ctx)
            else:
                agenda_md = "## Agenda\n" + "\n".join(f"- {p}" for p in app.discussion_points)

            app.last_agenda = agenda_md
            _write_text(os.path.join(_sess_dir(app.session_id), "agenda.md"), agenda_md)

            st.success("Agenda generated and saved to cache!")
            st.markdown("### Generated Agenda")
            st.markdown(f'<div class="card">{agenda_md}</div>', unsafe_allow_html=True)
            st.download_button("Download Agenda (.md)", data=agenda_md.encode("utf-8"),
                               file_name="agenda.md", mime="text/markdown", use_container_width=True)

# ---------------- TRACKING ----------------
with tab_track:
    st.markdown("### 🎥 Meeting Tracking & Agenda Resolution")
    if not (app.session_id and app.discussion_points):
        st.warning("No agenda points available. Please finish **Pre-Meeting**.")
    else:
        v = st.file_uploader("Upload meeting video or audio", type=["mp4", "mov", "avi", "wav", "mp3", "m4a"])
        trans_bar = st.progress(0.0, text="Waiting for media...")
        eta_txt = st.empty()

        if v:
            if v.type.startswith("video"):
                st.video(v)
            else:
                st.audio(v)

            if st.button("Transcribe & Analyze", type="primary", use_container_width=True):
                suffix = os.path.splitext(v.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_v:
                    tmp_v.write(v.getbuffer())
                    media_path = tmp_v.name

                audio_path = extract_audio_to_wav(media_path)
                safe_unlink(media_path)

                def _progress_cb(p: float, eta: float):
                    trans_bar.progress(p, text=f"Transcribing {int(p*100)}%")
                    eta_txt.markdown(f"<div class='small'>ETA ~ {int(eta)}s</div>", unsafe_allow_html=True)

                try:
                    transcript = transcribe_long_audio(audio_path, progress_cb=_progress_cb)
                    transcript = dedupe_lines(transcript)
                    app.last_transcript = transcript
                    _write_text(os.path.join(_sess_dir(app.session_id), "transcript.txt"), transcript)
                    trans_bar.progress(1.0, text="Transcription complete")
                finally:
                    safe_unlink(audio_path)

                resolved, unresolved = analyze_agenda_resolution(app.discussion_points, app.last_transcript or "")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### ✅ Resolved")
                    if resolved: st.success("\n".join(f"• {r}" for r in resolved))
                    else: st.write("—")
                with c2:
                    st.markdown("#### ⏳ Unresolved")
                    if unresolved: st.warning("\n".join(f"• {u}" for u in unresolved))
                    else: st.write("—")

                with st.expander("Show Transcript"):
                    st.write(app.last_transcript or "")

# ---------------- SUMMARY ----------------
with tab_summary:
    st.markdown("### 📝 Post-Meeting Summary")
    if not (app.session_id and (app.faiss_store or app.bm25)):
        st.warning("Please complete **Pre-Meeting** so RAG context is available.")
    else:
        v2 = st.file_uploader("Upload meeting media (optional)", type=["mp4","mov","avi","wav","mp3","m4a"], key="post_vid")
        query = st.text_input("Focus query",
                              value="Key discussion topics, decisions made, action items with owners",
                              help="What do you want the summary to emphasize?")

        if st.button("Generate Summary", type="primary", use_container_width=True):
            transcript = app.last_transcript or ""
            if v2 and not transcript:
                suffix = os.path.splitext(v2.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_v:
                    tmp_v.write(v2.getbuffer()); media_path = tmp_v.name
                audio_path = extract_audio_to_wav(media_path)
                safe_unlink(media_path)
                trans_bar = st.progress(0.0, text="Transcribing...")
                eta_txt = st.empty()
                def _progress_cb(p: float, eta: float):
                    trans_bar.progress(p, text=f"Transcribing {int(p*100)}%")
                    eta_txt.markdown(f"<div class='small'>ETA ~ {int(eta)}s</div>", unsafe_allow_html=True)
                try:
                    transcript = transcribe_long_audio(audio_path, progress_cb=_progress_cb)
                    transcript = dedupe_lines(transcript)
                    app.last_transcript = transcript
                    _write_text(os.path.join(_sess_dir(app.session_id), "transcript.txt"), transcript)
                finally:
                    safe_unlink(audio_path)

            if not transcript.strip():
                st.error("No transcript available. Upload media in **Tracking** or here.")
            else:
                if not app.last_doc_text:
                    extracted_path = os.path.join(_sess_dir(app.session_id), "extracted.txt")
                    app.last_doc_text = _read_text(extracted_path)
                app.chunks = build_chunks(app.last_doc_text)

                q = (", ".join(app.discussion_points or []) + " " + query).strip()
                ctx = select_context(app.faiss_store, app.bm25, app.chunks or [], q, k=4)

                llm = maybe_llm(max_tokens=450, temperature=TEMPERATURE)
                if CHAT_OK and llm:
                    chain = LLMChain(prompt=SUMMARY_PROMPT_JSON, llm=llm)
                    obj = run_json(
                        chain,
                        ctx=truncate_tokens(ctx, MAX_INPUT_TOKENS//2),
                        transcript=truncate_tokens(transcript, MAX_INPUT_TOKENS//2),
                        query=query
                    )
                else:
                    # Heuristic fallback summary
                    obj = {
                        "topics": list({t.strip() for t in (", ".join(app.discussion_points or [])).split(",") if t.strip()}),
                        "decisions": [],
                        "action_items": []
                    }

                app.last_summary = obj
                _write_json(os.path.join(_sess_dir(app.session_id), "summary.json"), obj)

                st.markdown("### Summary (JSON)")
                st.code(json.dumps(obj, indent=2))
                st.download_button("Download Summary (JSON)",
                                   data=json.dumps(obj, indent=2).encode("utf-8"),
                                   file_name="post_meeting_summary.json",
                                   mime="application/json",
                                   use_container_width=True)

# ---------------- NOTES ----------------
if not st.session_state.OPENAI_API_KEY:
    st.info("No OpenAI API key set. Agenda/Summary will use robust fallbacks. Embeddings default to MiniLM (CPU).")
