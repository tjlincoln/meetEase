# =======================
# MeetEase — NO-DB EDITION (with robust ffmpeg handling)
# Fully local (session + JSON files), no MySQL/Postgres required
# =======================
from __future__ import annotations

# stdlib
import os
import io
import re
import csv
import json
import time
import math
import gc
import hashlib
import tempfile
import subprocess
import shutil
import warnings
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from datetime import date, datetime

# third-party
import streamlit as st                       # UI

from PIL import Image                        # OCR helpers
import numpy as np
import pytesseract
import fitz                                   # PyMuPDF
import docx                                   # python-docx
import cv2

import tiktoken                               # token counting
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from pydub import AudioSegment                # audio I/O (fallback)
from faster_whisper import WhisperModel       # fast STT

# Optional, portable ffmpeg provider
try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None

# =======================
# CONFIG / ENV
# =======================
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL         = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL   = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
EMBED_MODE           = os.getenv("MEETEASE_EMBED_MODE", "minilm").lower()  # "minilm" or "openai"
MINILM_MODEL_NAME    = os.getenv("MINILM_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
TEMPERATURE          = float(os.getenv("MEETEASE_TEMPERATURE", "0.2"))
MAX_INPUT_TOKENS     = int(os.getenv("MEETEASE_MAX_INPUT_TOKENS", "6000"))

WHISPER_MODEL        = os.getenv("WHISPER_MODEL", "medium")
TESSERACT_PATH_WIN   = os.getenv("TESSERACT_PATH_WIN", "")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
CACHE_DIR  = os.getenv("CACHE_DIR", "cache")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)

LOCAL_DB_PATH = os.path.join(CACHE_DIR, "local_store.json")

# =======================
# STREAMLIT UI SHELL
# =======================
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
st.markdown('<div class="subtle">Prepare, run, and summarize meetings with AI — fully local, no database.</div>', unsafe_allow_html=True)
st.write("")

# =======================
# LOCAL JSON “STORE”
# =======================
def _load_store() -> Dict:
    if not os.path.isfile(LOCAL_DB_PATH):
        return {
            "meetings": [],          # {id, title, meeting_date, created_at}
            "documents": [],         # {id, meeting_id, name, mime, hash_key, text, created_at}
            "doc_chunks": [],        # {document_id, chunk_index, text, hash_key}
            "agendas": [],           # {meeting_id, agenda_text, created_at}
            "transcripts": [],       # {meeting_id, audio_hash, transcript, created_at}
            "summaries": []          # {meeting_id, query_text, summary_text(json), created_at}
        }
    with open(LOCAL_DB_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {
                "meetings": [], "documents": [], "doc_chunks": [],
                "agendas": [], "transcripts": [], "summaries": []
            }

def _save_store(store: Dict):
    tmp = LOCAL_DB_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)
    os.replace(tmp, LOCAL_DB_PATH)

def _next_id(items: List[Dict]) -> int:
    return (max((x.get("id", 0) for x in items), default=0) + 1)

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def settings_hash(d: Dict) -> str:
    s = json.dumps(d, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()

# =======================
# TOKEN / EMBEDDINGS / SPLITTER
# =======================
@st.cache_resource(show_spinner=False)
def token_encoder_cached():
    import tiktoken
    try: return tiktoken.encoding_for_model(OPENAI_MODEL)
    except Exception: return tiktoken.get_encoding("cl100k_base")

def truncate_tokens(text: str, max_tokens: int) -> str:
    enc = token_encoder_cached()
    ids = enc.encode(text or "")
    return enc.decode(ids[:max_tokens])

@st.cache_resource(show_spinner=False)
def get_embeddings_cached():
    if EMBED_MODE == "openai" and OPENAI_API_KEY:
        return OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    return HuggingFaceEmbeddings(model_name=MINILM_MODEL_NAME, encode_kwargs={"normalize_embeddings": True})

@st.cache_resource(show_spinner=False)
def get_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=160)

def build_chunks(text: str) -> List[str]:
    splitter = get_splitter()
    chunks = splitter.split_text(text or "")
    return chunks if chunks else [text or ""]

# =======================
# OCR / FILE EXTRACTORS
# =======================
if TESSERACT_PATH_WIN:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH_WIN

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
            raw = pil_ocr(proc)
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

# =======================
# LLM / PROMPTS
# =======================
def dedupe_lines(text: str) -> str:
    seen, out = set(), []
    for line in [l.strip() for l in (text or "").splitlines() if l.strip()]:
        key = re.sub(r"\W+","", line.lower())
        if key in seen: continue
        seen.add(key); out.append(line)
    return "\n".join(out)

AGENDA_PROMPT = PromptTemplate(
    input_variables=["discussion_points", "context"],
    template=(
        "You are a project coordinator. Create a concise, well-structured meeting agenda.\n\n"
        "Discussion points:\n{discussion_points}\n\n"
        "Relevant context (from docs):\n{context}\n\n"
        "Return a professional agenda with sections, timings (optional), and logical flow."
    ),
)

SUMMARY_PROMPT_JSON = PromptTemplate(
    input_variables=["ctx", "transcript", "query"],
    template=(
        "Using the context and transcript, produce a crisp post-meeting summary with:\n"
        "1) Key Discussion Topics\n2) Decisions Made\n3) Action Items with Owners & due dates when stated\n\n"
        "Context:\n{ctx}\n\nTranscript:\n{transcript}\n\nQuery:\n{query}\n"
    ),
)

@st.cache_resource(show_spinner=False)
def maybe_llm(max_tokens=400, temperature=0.2):
    if not OPENAI_API_KEY:
        return None
    return ChatOpenAI(model=OPENAI_MODEL, temperature=temperature, max_tokens=max_tokens)

def run_json(chain: Optional[LLMChain], **kwargs) -> Dict:
    out = chain.run(**kwargs).strip() if chain else ""
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
    return {"Context":"(LLM unavailable)", "Decisions":[], "ActionItems":[], "Risks":[]}

# =======================
# STT (FASTER-WHISPER) + FFMPEG HANDLING
# =======================
def get_ffmpeg_path() -> Optional[str]:
    """Return a working ffmpeg path or None if unavailable."""
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    return None

@st.cache_resource(show_spinner=False)
def load_whisper(model_name: str):
    device = "cuda" if (os.getenv("CUDA_VISIBLE_DEVICES") not in (None, "", "-1")) else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(model_name, device=device, compute_type=compute_type)

def safe_tmp_path(suffix=".wav") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix); os.close(fd); return path

def safe_unlink(path: str, max_tries: int = 6, wait_s: float = 0.25):
    for i in range(max_tries):
        try:
            if os.path.exists(path): os.remove(path)
            return
        except PermissionError:
            time.sleep(wait_s * (i + 1)); gc.collect()
    try:
        if os.path.exists(path): os.remove(path)
    except Exception:
        pass

def extract_audio_to_wav(media_path: str) -> str:
    """Preferred: use ffmpeg (system or imageio-ffmpeg). Fallback to Pydub if absolutely needed."""
    out_wav = safe_tmp_path(".wav")
    ff = get_ffmpeg_path()
    if ff:
        subprocess.run(
            [ff, "-y", "-i", media_path, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", out_wav],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return out_wav

    # Final fallback (note: pydub still requires ffmpeg/ffprobe in many environments)
    AudioSegment.from_file(media_path).set_frame_rate(16000).set_channels(1).export(out_wav, format="wav")
    return out_wav

def transcribe_long_audio(audio_path: str, progress_cb=None) -> str:
    model = load_whisper(WHISPER_MODEL)
    segments, info = model.transcribe(
        audio_path,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=800)
    )
    text_parts, total_dur = [], (info.duration or 1.0)
    last_t = 0.0
    for seg in segments:
        text_parts.append(seg.text.strip())
        last_t = seg.end or last_t
        if progress_cb:
            p = min(1.0, last_t / total_dur)
            progress_cb(p, max(0.0, total_dur - last_t))
    final = " ".join(t for t in text_parts if t)
    final = re.sub(r"\s+", " ", final).strip()
    return dedupe_lines(final)

# =======================
# LOCAL STORE HELPERS
# =======================
def meeting_get_or_create_local(title: str, mdate: date) -> int:
    store = _load_store()
    for m in store["meetings"]:
        if m["title"] == title and m["meeting_date"] == str(mdate):
            return m["id"]
    mid = _next_id(store["meetings"])
    store["meetings"].append({
        "id": mid,
        "title": title,
        "meeting_date": str(mdate),
        "created_at": datetime.utcnow().isoformat()
    })
    _save_store(store)
    return mid

def document_get_or_create_local(meeting_id: int, name: str, mime: str, content: bytes) -> Tuple[int, str]:
    h = sha256_bytes(content)
    store = _load_store()
    for d in store["documents"]:
        if d["hash_key"] == h and d["meeting_id"] == meeting_id:
            return d["id"], h
    did = _next_id(store["documents"])
    store["documents"].append({
        "id": did,
        "meeting_id": meeting_id,
        "name": name,
        "mime": mime,
        "hash_key": h,
        "text": "",
        "created_at": datetime.utcnow().isoformat()
    })
    _save_store(store)
    return did, h

def document_update_text_local(doc_id: int, text: str):
    store = _load_store()
    for d in store["documents"]:
        if d["id"] == doc_id:
            d["text"] = text
            break
    _save_store(store)

def chunks_upsert_local(doc_id: int, chunks: List[str]):
    store = _load_store()
    store["doc_chunks"] = [c for c in store["doc_chunks"] if c["document_id"] != doc_id]
    for i, c in enumerate(chunks):
        hk = sha256_bytes(c.encode("utf-8"))
        store["doc_chunks"].append({
            "document_id": doc_id,
            "chunk_index": i,
            "text": c,
            "hash_key": hk
        })
    _save_store(store)

def agenda_insert_local(meeting_id: int, agenda_text: str):
    store = _load_store()
    store["agendas"].append({
        "meeting_id": meeting_id,
        "agenda_text": agenda_text,
        "created_at": datetime.utcnow().isoformat()
    })
    _save_store(store)

def transcript_upsert_local(meeting_id: int, audio_hash: str, transcript_text: str):
    store = _load_store()
    found = False
    for t in store["transcripts"]:
        if t["meeting_id"] == meeting_id and t["audio_hash"] == audio_hash:
            t["transcript"] = transcript_text
            found = True
            break
    if not found:
        store["transcripts"].append({
            "meeting_id": meeting_id,
            "audio_hash": audio_hash,
            "transcript": transcript_text,
            "created_at": datetime.utcnow().isoformat()
        })
    _save_store(store)

def summary_insert_local(meeting_id: int, query_text: str, summary_json: Dict):
    store = _load_store()
    store["summaries"].append({
        "meeting_id": meeting_id,
        "query_text": query_text,
        "summary_text": summary_json,
        "created_at": datetime.utcnow().isoformat()
    })
    _save_store(store)

def get_latest_document_text_local(meeting_id: int) -> str:
    store = _load_store()
    docs = [d for d in store["documents"] if d["meeting_id"] == meeting_id]
    if not docs: return ""
    docs.sort(key=lambda d: d.get("created_at",""), reverse=True)
    return docs[0].get("text","") or ""

# =======================
# RAG HELPERS (FAISS/BM25 on disk only)
# =======================
def faiss_save(store_obj: FAISS, path_dir: str):
    os.makedirs(path_dir, exist_ok=True)
    store_obj.save_local(path_dir)

def faiss_load(path_dir: str, embeddings):
    if not os.path.isdir(path_dir): return None
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

def build_and_persist_indices(doc_id: int, doc_hash: str, full_text: str, embed_mode: str, splitter_conf: Dict) -> Tuple[FAISS, BM25Okapi, List[str], str, str]:
    s_hash = settings_hash({
        "embed_mode": embed_mode,
        "model": MINILM_MODEL_NAME if embed_mode=="minilm" else OPENAI_EMBED_MODEL,
        "splitter": splitter_conf,
        "doc_hash": doc_hash,
    })
    faiss_dir = os.path.join(CACHE_DIR, f"faiss_{doc_id}_{s_hash[:8]}")
    bm25_path = os.path.join(CACHE_DIR, f"bm25_{doc_id}_{s_hash[:8]}.pkl")

    embeddings = get_embeddings_cached()
    chunks = build_chunks(full_text)

    store_obj = faiss_load(faiss_dir, embeddings)
    bm25 = bm25_load(bm25_path)
    if store_obj and bm25:
        chunks_upsert_local(doc_id, chunks)
        return store_obj, bm25, chunks, faiss_dir, bm25_path

    store_obj = FAISS.from_texts(chunks, embeddings)
    faiss_save(store_obj, faiss_dir)
    bm25 = BM25Okapi([c.split() for c in chunks])
    bm25_save(bm25, bm25_path)
    chunks_upsert_local(doc_id, chunks)
    return store_obj, bm25, chunks, faiss_dir, bm25_path

def select_context(store_obj: Optional[FAISS], bm25: Optional[BM25Okapi], chunks: List[str], query: str, k:int=4) -> str:
    ctx = ""
    if store_obj:
        try:
            docs = store_obj.similarity_search(query, k=k)
            ctx = "\n\n".join(d.page_content for d in docs)
        except Exception:
            pass
    if (not ctx) and bm25 and chunks:
        scores = bm25.get_scores(query.split())
        top_idx = np.argsort(scores)[-k:][::-1]
        ctx = "\n\n".join(chunks[i] for i in top_idx)
    return ctx

# =======================
# AGENDA RESOLUTION
# =======================
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
        else: unresolved.append(p)
    return resolved, unresolved

# =======================
# APP STATE
# =======================
@dataclass
class AppState:
    meeting_id: Optional[int] = None
    document_id: Optional[int] = None
    document_hash: Optional[str] = None
    discussion_points: Optional[List[str]] = None
    chunks: Optional[List[str]] = None
    faiss_store: Optional[FAISS] = None
    bm25: Optional[BM25Okapi] = None
    last_transcript: Optional[str] = None
    last_doc_text: Optional[str] = None

if "app" not in st.session_state:
    st.session_state.app = AppState()
app: AppState = st.session_state.app

# =======================
# TABS
# =======================
tab_pre, tab_agenda, tab_track, tab_summary = st.tabs(
    ["📄 Pre-Meeting", "📋 Agenda", "🎥 Tracking", "📝 Post-Summary"]
)

# -------- PRE-MEETING TAB --------
with tab_pre:
    st.markdown("### 📄 Pre-Meeting — Documents & Discussion Points")
    colA, colB = st.columns([1.15, 1])

    with colA:
        title = st.text_input("Meeting title", value="Weekly Sync")
        mdate = st.date_input("Meeting date", value=date.today())
        up = st.file_uploader("Upload a document (PDF, DOCX, or Image)", type=["pdf", "docx", "png", "jpg", "jpeg"])
        dpoints = st.text_area("Discussion points (comma-separated)", placeholder="Budget approval, Roadmap alignment, Risk review ...", height=120)

        # Diagnostics
        with st.expander("Diagnostics"):
            path = get_ffmpeg_path()
            st.write("ffmpeg:", path or "(not found)")
            if path:
                try:
                    out = subprocess.run([path, "-version"], capture_output=True, text=True, check=True)
                    st.code(out.stdout.strip()[:4000])
                except Exception as e:
                    st.write("ffmpeg -version error:", str(e)[:300])

        ocr_bar = st.progress(0.0, text="Waiting for document...")

        if st.button("Process & Build Context", type="primary", use_container_width=True):
            if not title.strip():
                st.warning("Please enter a meeting title."); st.stop()
            if not up or not dpoints.strip():
                st.warning("Please upload a document and add discussion points."); st.stop()

            app.meeting_id = meeting_get_or_create_local(title.strip(), mdate)

            raw = up.read()
            app.document_id, app.document_hash = document_get_or_create_local(app.meeting_id, up.name, up.type or "", raw)

            # Extract or reuse text
            store = _load_store()
            doc_rec = next((d for d in store["documents"] if d["id"] == app.document_id), None)
            text = (doc_rec or {}).get("text", "")

            if not text:
                ext = (up.name.split(".")[-1] or "").lower()
                if ext == "pdf":
                    ocr_bar.progress(0.2, text="Extracting PDF text...")
                    text = cached_extract_pdf_text(app.document_hash, raw)
                    ocr_bar.progress(1.0, text="PDF processed")
                elif ext == "docx":
                    ocr_bar.progress(0.3, text="Extracting from DOCX...")
                    text = cached_extract_docx_text(app.document_hash, raw)
                    ocr_bar.progress(1.0, text="DOCX extracted")
                elif ext in ("png","jpg","jpeg"):
                    ocr_bar.progress(0.3, text="OCR image...")
                    text = cached_extract_image_text(app.document_hash, raw)
                    ocr_bar.progress(1.0, text="Image OCR done")
                else:
                    st.error("Unsupported file type."); st.stop()
                document_update_text_local(app.document_id, text)
            else:
                ocr_bar.progress(1.0, text="Reusing cached text")

            app.last_doc_text = text

            # Build or load indices
            st.info("Building / loading indices...")
            splitter_conf = {"chunk_size":1400, "overlap":160}
            app.faiss_store, app.bm25, app.chunks, _, _ = build_and_persist_indices(
                app.document_id, app.document_hash or "", app.last_doc_text or "",
                EMBED_MODE, splitter_conf
            )

            app.discussion_points = [p.strip() for p in dpoints.split(",") if p.strip()]
            st.success("Context ready! Go to **Agenda** to generate an agenda.")

    with colB:
        st.markdown("#### Document Snapshot")
        if app.document_id:
            preview = app.last_doc_text or ""
            token_count = len(preview.split())
            st.markdown(f'<div class="card"><div class="kv"><b>Words</b> {token_count}</div><div class="kv"><b>Preview</b></div></div>', unsafe_allow_html=True)
            with st.expander("Show extracted text (first 1200 chars)"):
                st.write(preview[:1200] + ("..." if len(preview) > 1200 else ""))
        else:
            st.info("No document processed yet.")

# -------- AGENDA TAB --------
with tab_agenda:
    st.markdown("### 📋 Agenda Creation")
    if not (app.meeting_id and app.discussion_points and app.faiss_store):
        st.warning("Please complete **Pre-Meeting** first.")
    else:
        st.markdown("**Your discussion points**")
        st.write(", ".join(app.discussion_points))

        if st.button("Generate Agenda", type="primary", use_container_width=True):
            q = " ".join(app.discussion_points)
            ctx = select_context(app.faiss_store, app.bm25, app.chunks or [], q, k=4)
            llm = maybe_llm(max_tokens=400, temperature=TEMPERATURE)
            chain = LLMChain(prompt=AGENDA_PROMPT, llm=llm) if llm else None
            in_ctx = truncate_tokens(ctx, MAX_INPUT_TOKENS // 2)
            if chain:
                agenda_md = chain.run(discussion_points="\n- " + "\n- ".join(app.discussion_points), context=in_ctx)
            else:
                agenda_md = "## Agenda\n" + "\n".join(f"- {p}" for p in app.discussion_points)
            agenda_insert_local(app.meeting_id, agenda_md)
            st.success("Agenda generated and saved locally!")
            st.markdown("### Generated Agenda")
            st.markdown(f'<div class="card">{agenda_md}</div>', unsafe_allow_html=True)
            st.download_button("Download Agenda (.md)", data=agenda_md.encode("utf-8"),
                               file_name="agenda.md", mime="text/markdown", use_container_width=True)

# -------- TRACKING TAB --------
with tab_track:
    st.markdown("### 🎥 Meeting Tracking & Agenda Resolution")
    if not (app.meeting_id and app.discussion_points):
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
                    with open(audio_path, "rb") as f:
                        audio_hash = sha256_bytes(f.read())
                    transcript_upsert_local(app.meeting_id, audio_hash, transcript)
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

# -------- SUMMARY TAB --------
with tab_summary:
    st.markdown("### 📝 Post-Meeting Summary")
    if not (app.meeting_id and app.faiss_store):
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
                    with open(audio_path, "rb") as f:
                        audio_hash = sha256_bytes(f.read())
                    transcript_upsert_local(app.meeting_id, audio_hash, transcript)
                finally:
                    safe_unlink(audio_path)

            if not (app.last_transcript or "").strip():
                st.error("No transcript available. Upload media in **Tracking** or here.")
            else:
                if not app.last_doc_text:
                    app.last_doc_text = get_latest_document_text_local(app.meeting_id)
                app.chunks = build_chunks(app.last_doc_text)

                q = (", ".join(app.discussion_points or []) + " " + query).strip()
                ctx = select_context(app.faiss_store, app.bm25, app.chunks or [], q, k=4)
                llm = maybe_llm(max_tokens=450, temperature=TEMPERATURE)
                chain = LLMChain(prompt=SUMMARY_PROMPT_JSON, llm=llm) if llm else None
                obj = run_json(chain,
                               ctx=truncate_tokens(ctx, MAX_INPUT_TOKENS//2),
                               transcript=truncate_tokens(app.last_transcript, MAX_INPUT_TOKENS//2),
                               query=query)

                summary_insert_local(app.meeting_id, query, obj)

                st.markdown("### Summary (JSON)")
                st.code(json.dumps(obj, indent=2))
                st.download_button("Download Summary (JSON)",
                                   data=json.dumps(obj, indent=2).encode("utf-8"),
                                   file_name="post_meeting_summary.json",
                                   mime="application/json",
                                   use_container_width=True)

# ======= FINAL NOTE =======
if not OPENAI_API_KEY:
    st.info("No OPENAI_API_KEY set. Agenda/Summary will use robust fallbacks (JSON heuristic). Embeddings default to MiniLM CPU.")
