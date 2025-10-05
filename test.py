import os
import sys
import traceback
from datetime import datetime, date
from typing import Optional, List, Dict

import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor

# -----------------------------
# Configuration helpers
# -----------------------------
def get_db_cfg() -> Dict[str, str]:
    """
    Pull DB config from st.secrets['postgres'] first;
    fall back to environment variables if running locally without secrets.
    """
    cfg = {}
    if "postgres" in st.secrets:
        cfg = dict(st.secrets["postgres"])
    else:
        cfg = {
            "host": os.getenv("PGHOST", ""),
            "port": os.getenv("PGPORT", "6543"),
            "dbname": os.getenv("PGDATABASE", "postgres"),
            "user": os.getenv("PGUSER", "postgres"),
            "password": os.getenv("PGPASSWORD", ""),
            "sslmode": os.getenv("PGSSLMODE", "require"),
        }
    # Sanity defaults
    cfg["port"] = int(cfg.get("port", 6543) or 6543)
    cfg["sslmode"] = cfg.get("sslmode", "require") or "require"
    return cfg


def db_conn():
    """
    Create a psycopg2 connection using separate parameters.
    This avoids URL parsing issues when the password contains special characters like '@'.
    """
    cfg = get_db_cfg()
    # Note: connect_timeout keeps the app from hanging if host/port are wrong.
    return psycopg2.connect(
        host=cfg["host"],
        port=cfg["port"],
        dbname=cfg["dbname"],
        user=cfg["user"],
        password=cfg["password"],
        sslmode=cfg["sslmode"],
        connect_timeout=10,
        cursor_factory=RealDictCursor,
    )


# -----------------------------
# DB bootstrap (idempotent)
# -----------------------------
DDL_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS meetings (
    id          BIGSERIAL PRIMARY KEY,
    title       VARCHAR(255) NOT NULL,
    meeting_date DATE        NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Ensure (title, meeting_date) is unique so we can UPSERT cleanly
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM   pg_indexes
        WHERE  schemaname = 'public'
        AND    indexname  = 'uniq_meetings_title_date'
    ) THEN
        CREATE UNIQUE INDEX uniq_meetings_title_date
        ON meetings (title, meeting_date);
    END IF;
END
$$;
"""

UPSERT_MEETING = """
INSERT INTO meetings (title, meeting_date)
VALUES (%s, %s)
ON CONFLICT (title, meeting_date)
DO UPDATE SET title = EXCLUDED.title
RETURNING id;
"""

SELECT_RECENT = """
SELECT id, title, meeting_date, created_at
FROM meetings
ORDER BY meeting_date DESC, id DESC
LIMIT %s;
"""

DELETE_BY_ID = """
DELETE FROM meetings WHERE id = %s;
"""


def ensure_schema():
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(DDL_CREATE_TABLES)
        conn.commit()


def meeting_get_or_create(title: str, mdate: date) -> int:
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(UPSERT_MEETING, (title.strip(), mdate))
        row = cur.fetchone()
        return int(row["id"])


def list_recent_meetings(limit: int = 20) -> List[Dict]:
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(SELECT_RECENT, (limit,))
        return cur.fetchall() or []


def delete_meeting(mid: int) -> int:
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(DELETE_BY_ID, (mid,))
        return cur.rowcount


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="MeetEase — DB Test", page_icon="🗂", layout="centered")
st.title("🗂 MeetEase — Supabase Postgres (Pooler)")

st.caption(
    "This page verifies your Streamlit ↔ Supabase (Pooler) connection, "
    "creates the schema if missing, and lets you quickly add/list meetings."
)

with st.expander("Database connection & environment check", expanded=True):
    cfg = get_db_cfg()
    st.write("**DB target**")
    st.code(
        f"host={cfg.get('host')}  port={cfg.get('port')}  dbname={cfg.get('dbname')}  user={cfg.get('user')}  sslmode={cfg.get('sslmode')}",
        language="bash",
    )
    _ok = False
    try:
        with db_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT version();")
            ver = cur.fetchone()
            st.success("Connected successfully ✔")
            st.write("PostgreSQL:", ver.get("version"))
            _ok = True
    except Exception as e:
        st.error("❌ Connection failed. Check host/port, user/password, and sslmode.")
        st.exception(e)

    st.divider()
    colA, colB = st.columns(2)
    with colA:
        if st.button("Ensure schema (create tables if missing)", disabled=not _ok, use_container_width=True):
            try:
                ensure_schema()
                st.success("Schema ensured (idempotent).")
            except Exception as e:
                st.error("Failed to ensure schema.")
                st.exception(e)

    with colB:
        st.write("")


st.header("Create or get meeting")
with st.form("create_meeting"):
    title = st.text_input("Title", value="Weekly Sync")
    mdate = st.date_input("Meeting date", value=date.today())
    submitted = st.form_submit_button("Save / Upsert")
    if submitted:
        try:
            ensure_schema()
            mid = meeting_get_or_create(title, mdate)
            st.success(f"Saved. Meeting ID = {mid}")
        except Exception as e:
            st.error("Could not save meeting.")
            st.exception(e)

st.header("Recent meetings")
limit = st.slider("How many to show", 5, 100, 20, step=5)
try:
    rows = list_recent_meetings(limit)
    if not rows:
        st.info("No meetings yet.")
    else:
        for r in rows:
            with st.container(border=True):
                st.write(f"**ID**: {r['id']}  |  **Title**: {r['title']}")
                st.write(f"**Date**: {r['meeting_date']}  |  **Created**: {r['created_at']}")
                del_col1, del_col2 = st.columns([1, 5])
                with del_col1:
                    if st.button("Delete", key=f"del-{r['id']}"):
                        try:
                            n = delete_meeting(r["id"])
                            if n:
                                st.success(f"Deleted meeting {r['id']}. Refresh to see changes.")
                            else:
                                st.warning("Nothing deleted (ID not found).")
                        except Exception as e:
                            st.error("Delete failed.")
                            st.exception(e)
except Exception as e:
    st.error("Failed to list meetings.")
    st.exception(e)
