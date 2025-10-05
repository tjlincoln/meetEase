# app.py — MeetEase (FAST, single-file, pooler-friendly, no DDL)
# --------------------------------------------------------------------------------------
# Assumes DB tables already exist. No CREATE TABLE statements here.
# - Supabase: tries Pooler first (recommended on Streamlit Cloud), then direct.
# - OCR: PyMuPDF + Tesseract (optional)
# - Indexing: FAISS (if available) + BM25, with disk caching
# - STT: faster-whisper (CTranslate2) with ffmpeg/pydub fallback
# - LLM: OpenAI optional; graceful fallbacks when not set
# --------------------------------------------------------------------------------------

from __future__ import annotations

import os, io, re, csv, json, time, math, pickle, hashlib, tempfile, warnings, gc, subprocess, shutil, sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from datetime import date, datetime

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================== CONFIG ===============================
# Project ref from your Supabase URL: https://mvnvxfuiyggatlakgrbr.supabase.co
PROJECT_REF = "mvnvxfuiyggatlakgrbr"

# --- Pooler (recommended on Streamlit Cloud) ---
SB_HOST_POOLER = os.getenv("SB_HOST_POOLER", "aws-1-ap-southeast-1.pooler.supabase.com")
SB_PORT_POOLER = int(os.getenv("SB_PORT_POOLER", "6543"))
SB_USER_POOLER = os.getenv("SB_USER_POOLER", f"postgres.{PROJECT_REF}")  # NOTE: user.projectref

# --- Direct (works well locally/servers with public egress) ---
SB_HOST_DIRECT = os.getenv("SB_HOST_DIRECT", f"db.{PROJECT_REF}.supabase.co")
SB_PORT_DIRECT = int(os.getenv("SB_PORT_DIRECT", "5432"))
SB_USER_DIRECT = os.getenv("SB_USER_DIRECT", "postgres")

# Shared
SUPABASE_DB       = os.getenv("SUPABASE_DATABASE", "postgres")
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD", "REPLACE_ME")  # <--- put your DB password or use env/secrets
USE_POOLER_FIRST  = os.getenv("USE_POOLER_FIRST", "true").lower() in ("1","true","yes")

# OpenAI (optional)
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "").strip()
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

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # used by langchain_openai

# ============================== UI ===============================
import streamlit as st
st.set_page_config(page_title="MeetEase — Meeting Management", page_icon="🎯", layout="wide")
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

st.markdown('<div class="big-title">🤖 MeetEase — Meeting Management</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Prepare, run, and summarize meetings with AI assistance.</div>', unsafe_allow_html=True)
st.write("")

# ============================== DB ===============================
import psycopg2
from psycopg2.extras import RealDictCursor

def _try_connect(host, port, user, dbname, pwd, label):
    try:
        conn = psycopg2.connect(
            host=host, port=port, user=user, password=pwd, dbname=dbname,
            sslmode="require", cursor_factory=RealDictCursor, connect_timeout=10
        )
        return conn, None
    except Exception as e:
        return None, f"{label}: {e}"

def db_conn():
    # Prefer pooler on cloud; fall back to direct if needed
    order = [
        ("POOLER", SB_HOST_POOLER, SB_PORT_POOLER, SB_USER_POOLER),
        ("DIRECT", SB_HOST_DIRECT, SB_PORT_DIRECT, SB_USER_DIRECT),
    ]
    if not USE_POOLER_FIRST:
        order.reverse()

    last_err = None
    for label, host, port, user in order:
        conn, err = _try_connect(host, port, user, SUPABASE_DB, SUPABASE_PASSWORD, label)
        if conn: 
            st.session_state["_db_mode"] = label
            st.session_state["_db_host"] = host
            st.session_state["_db_user"] = user
            return conn
        last_err = err
    raise RuntimeError(f"DB connection failed. Last error: {last_err}")

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
        # Soft fallback if tiktoken not installed
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
    if EMBED_MODE == "openai" and OPENAI_API_KEY and OPENAI_EMB_OK:
        return OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    if HF_OK:
        return HuggingFaceEmbeddings(model_name=MINILM_MODEL_NAME, encode_kwargs={"normalize_embeddings": True})
    # Minimal fallback: an object with a .embed_documents interface
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
    if not (OPENAI_API_KEY and CHAT_OK):
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
        # fallback to pydub (requires ffmpeg in PATH usually; but some hosts provide codecs)
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

# ============================== DB HELPERS ===============================
def meeting_get_or_create(title: str, mdate: date) -> int:
    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM meetings WHERE title=%s AND meeting_date=%s", (title, mdate))
            row = cur.fetchone()
            if row: return row["id"]
            cur.execute("INSERT INTO meetings (title, meeting_date) VALUES (%s, %s) RETURNING id", (title, mdate))
            new_id = cur.fetchone()["id"]
            conn.commit()
            return new_id
    finally:
        conn.close()

def document_get_or_create(meeting_id: int, name: str, mime: str, content: bytes) -> Tuple[int, str]:
    h = sha256_bytes(content)
    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM documents WHERE hash_key=%s AND meeting_id=%s", (h, meeting_id))
            r = cur.fetchone()
            if r: return r["id"], h
            cur.execute(
                "INSERT INTO documents (meeting_id, name, mime, hash_key) VALUES (%s,%s,%s,%s) RETURNING id",
                (meeting_id, name, mime, h)
            )
            new_id = cur.fetchone()["id"]
            conn.commit()
            return new_id, h
    finally:
        conn.close()

def document_update_text(doc_id: int, text: str):
    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE documents SET text=%s WHERE id=%s", (text, doc_id))
            conn.commit()
    finally:
        conn.close()

def chunks_upsert(doc_id: int, chunks: List[str]):
    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM doc_chunks WHERE document_id=%s", (doc_id,))
            for i, c in enumerate(chunks):
                hk = sha256_bytes(c.encode("utf-8"))
                cur.execute(
                    "INSERT INTO doc_chunks (document_id, chunk_index, text, hash_key) VALUES (%s,%s,%s,%s)",
                    (doc_id, i, c, hk)
                )
            conn.commit()
    finally:
        conn.close()

def indices_get(doc_id: int):
    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM indices WHERE document_id=%s", (doc_id,))
            return cur.fetchone()
    finally:
        conn.close()

def indices_upsert(doc_id: int, doc_hash: str, bm25_path: str, embed_index_path: str, embed_model: str):
    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM indices WHERE document_id=%s", (doc_id,))
            row = cur.fetchone()
            if row:
                cur.execute(
                    "UPDATE indices SET doc_hash=%s, bm25_path=%s, embed_index_path=%s, embed_model=%s WHERE id=%s",
                    (doc_hash, bm25_path, embed_index_path, embed_model, row["id"])
                )
            else:
                cur.execute(
                    "INSERT INTO indices (document_id, doc_hash, bm25_path, embed_index_path, embed_model) VALUES (%s,%s,%s,%s,%s)",
                    (doc_id, doc_hash, bm25_path, embed_index_path, embed_model)
                )
            conn.commit()
    finally:
        conn.close()

def agenda_insert(meeting_id: int, agenda_text: str):
    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO agendas (meeting_id, agenda_text) VALUES (%s,%s)", (meeting_id, agenda_text))
            conn.commit()
    finally:
        conn.close()

def transcript_upsert(meeting_id: int, audio_hash: str, transcript_text: str):
    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM transcripts WHERE meeting_id=%s AND audio_hash=%s", (meeting_id, audio_hash))
            r = cur.fetchone()
            if r:
                cur.execute("UPDATE transcripts SET transcript=%s WHERE id=%s", (transcript_text, r["id"]))
            else:
                cur.execute("INSERT INTO transcripts (meeting_id, audio_hash, transcript) VALUES (%s,%s,%s)",
                            (meeting_id, audio_hash, transcript_text))
            conn.commit()
    finally:
        conn.close()

def summary_insert(meeting_id: int, query_text: str, summary_json: Dict):
    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO summaries (meeting_id, query_text, summary_text) VALUES (%s,%s,%s)",
                        (meeting_id, query_text, json.dumps(summary_json, ensure_ascii=False)))
            conn.commit()
    finally:
        conn.close()

# ============================== INDEX BUILD/LOAD ===============================
def build_and_persist_indices(doc_id: int, doc_hash: str, full_text: str, embed_mode: str, splitter_conf: Dict):
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
                store = None  # FAISS failed; continue with BM25

    bm25 = bm25_load(bm25_path)
    if bm25 is None:
        bm25 = BM25Okapi([c.split() for c in chunks])
        bm25_save(bm25, bm25_path)

    chunks_upsert(doc_id, chunks)
    indices_upsert(
        doc_id, doc_hash, bm25_path, faiss_dir if store else "",
        ("openai:"+OPENAI_EMBED_MODEL) if EMBED_MODE=="openai" else ("hf:"+MINILM_MODEL_NAME)
    )
    return store, bm25, chunks, faiss_dir, bm25_path

def try_load_indices_with_settings(doc_id: int):
    rec = indices_get(doc_id)
    if not rec: return None, None, None
    embeddings = get_embeddings_cached()
    store = faiss_load(rec.get("embed_index_path",""), embeddings) if rec.get("embed_index_path") else None
    bm25  = bm25_load(rec.get("bm25_path","")) if rec.get("bm25_path") else None
    return store, bm25, None

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
    st.caption("Use this section to verify environment on Streamlit Cloud / server.")
    info_cols = st.columns(2)
    with info_cols[0]:
        st.write("**Python**", sys.version.split()[0])
        st.write("**Platform**", sys.platform)
        st.write("**ffmpeg**", shutil.which("ffmpeg") or "not-found")
        st.write("**tesseract**", shutil.which("tesseract") or (TESSERACT_PATH_WIN or "not-found"))
        st.write("**faster-whisper**", "OK" if FW_OK else "missing")
        st.write("**FAISS**", "OK" if FAISS_AVAILABLE else "missing (BM25 fallback)")
        st.write("**HF Embeddings**", "OK" if HF_OK else "missing")
        st.write("**OpenAI (LLM/Emb)**", "enabled" if OPENAI_API_KEY else "disabled")
    with info_cols[1]:
        try:
            conn = db_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                _ = cur.fetchone()
            conn.close()
            st.success(f"DB OK via {st.session_state.get('_db_mode','?')} → {st.session_state.get('_db_host','?')}")
            st.write(f"DB user: {st.session_state.get('_db_user','?')}")
        except Exception as e:
            st.error(f"DB connection failed: {e}")

# ============================== APP STATE ===============================
@dataclass
class AppState:
    meeting_id: Optional[int] = None
    document_id: Optional[int] = None
    document_hash: Optional[str] = None
    discussion_points: Optional[List[str]] = None
    chunks: Optional[List[str]] = None
    faiss_store: Optional[object] = None
    bm25: Optional[BM25Okapi] = None
    last_transcript: Optional[str] = None
    last_doc_text: Optional[str] = None

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

            app.meeting_id = meeting_get_or_create(title.strip(), mdate)

            raw = up.read()
            app.document_id, app.document_hash = document_get_or_create(app.meeting_id, up.name, up.type or "", raw)

            # Extract or reuse cached text from DB
            conn = db_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT text FROM documents WHERE id=%s", (app.document_id,))
                row = cur.fetchone()
            conn.close()
            text = (row or {}).get("text")

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
                document_update_text(app.document_id, text)
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

# ---------------- AGENDA ----------------
with tab_agenda:
    st.markdown("### 📋 Agenda Creation")
    if not (app.meeting_id and app.discussion_points and (app.faiss_store or app.bm25)):
        st.warning("Please complete **Pre-Meeting** first.")
    else:
        st.markdown("**Your discussion points**")
        st.write(", ".join(app.discussion_points))

        if st.button("Generate Agenda", type="primary", use_container_width=True):
            q = " ".join(app.discussion_points)
            ctx = select_context(app.faiss_store, app.bm25, app.chunks or [], q, k=4)
            llm = maybe_llm(max_tokens=400, temperature=TEMPERATURE)
            if CHAT_OK and llm:
                from langchain.chains import LLMChain
                chain = LLMChain(prompt=AGENDA_PROMPT, llm=llm)
                in_ctx = truncate_tokens(ctx, MAX_INPUT_TOKENS // 2)
                agenda_md = chain.run(discussion_points="\n- " + "\n- ".join(app.discussion_points), context=in_ctx)
            else:
                agenda_md = "## Agenda\n" + "\n".join(f"- {p}" for p in app.discussion_points)

            agenda_insert(app.meeting_id, agenda_md)
            st.success("Agenda generated and saved!")
            st.markdown("### Generated Agenda")
            st.markdown(f'<div class="card">{agenda_md}</div>', unsafe_allow_html=True)
            st.download_button("Download Agenda (.md)", data=agenda_md.encode("utf-8"),
                               file_name="agenda.md", mime="text/markdown", use_container_width=True)

# ---------------- TRACKING ----------------
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
                    transcript_upsert(app.meeting_id, audio_hash, transcript)
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
    if not (app.meeting_id and (app.faiss_store or app.bm25)):
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
                    transcript_upsert(app.meeting_id, audio_hash, transcript)
                finally:
                    safe_unlink(audio_path)

            if not transcript.strip():
                st.error("No transcript available. Upload media in **Tracking** or here.")
            else:
                if not app.last_doc_text:
                    conn = db_conn()
                    with conn.cursor() as cur:
                        cur.execute("SELECT text FROM documents WHERE meeting_id=%s ORDER BY created_at DESC LIMIT 1", (app.meeting_id,))
                        row = cur.fetchone()
                    conn.close()
                    app.last_doc_text = (row or {}).get("text") or ""
                app.chunks = build_chunks(app.last_doc_text)

                q = (", ".join(app.discussion_points or []) + " " + query).strip()
                ctx = select_context(app.faiss_store, app.bm25, app.chunks or [], q, k=4)

                llm = maybe_llm(max_tokens=450, temperature=TEMPERATURE)
                if CHAT_OK and llm:
                    from langchain.chains import LLMChain
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

                summary_insert(app.meeting_id, query, obj)

                st.markdown("### Summary (JSON)")
                st.code(json.dumps(obj, indent=2))
                st.download_button("Download Summary (JSON)",
                                   data=json.dumps(obj, indent=2).encode("utf-8"),
                                   file_name="post_meeting_summary.json",
                                   mime="application/json",
                                   use_container_width=True)

# ---------------- NOTES ----------------
if not OPENAI_API_KEY:
    st.info("No OPENAI_API_KEY set. Agenda/Summary will use robust fallbacks. Embeddings default to MiniLM (CPU).")
