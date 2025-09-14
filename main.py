import streamlit as st
import os
import whisper
import numpy as np
import pymysql
from moviepy.editor import VideoFileClip
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI as LangchainOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import docx
import tiktoken
import time

# -----------------------------
# SETUP
# -----------------------------
os.environ['OPENAI_API_KEY'] = "sk-xxxx"  # replace with your key

# MySQL connection setup
def get_db_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="root",
        database="meetingdb",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )

# -----------------------------
# FILE HANDLING FUNCTIONS
# -----------------------------
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def extract_text_from_doc(doc_file):
    doc = docx.Document(doc_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

# -----------------------------
# TOKEN UTILS
# -----------------------------
def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def truncate_text_tokens(text, max_tokens):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        return encoding.decode(tokens[:max_tokens])
    return text

# -----------------------------
# VECTOR DB
# -----------------------------
def create_vector_db(text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts, embeddings)

# -----------------------------
# AGENDA CREATION
# -----------------------------
def generate_agenda_with_discussion_points_rag(discussion_points, vectorstore):
    llm = LangchainOpenAI(temperature=0.5, max_tokens=400)
    relevant_docs = vectorstore.similarity_search(" ".join(discussion_points), k=5)
    relevant_text = "\n".join([doc.page_content for doc in relevant_docs])

    prompt_template = """
    You are a helpful assistant that organizes meeting agendas.
    Discussion points: {discussion_points}
    Relevant documents: {relevant_text}
    Organize into a structured agenda with related topics grouped.
    """

    truncated_relevant_text = truncate_text_tokens(relevant_text, 3000)

    prompt = PromptTemplate(
        input_variables=["discussion_points", "relevant_text"],
        template=prompt_template
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    return llm_chain.run(
        discussion_points=discussion_points,
        relevant_text=truncated_relevant_text
    )

# -----------------------------
# AUDIO + TRANSCRIPTION
# -----------------------------
def extract_audio_from_video(video_file):
    video_clip = VideoFileClip(video_file.name)
    audio_file = "temp_audio.wav"
    video_clip.audio.write_audiofile(audio_file, logger=None)
    video_clip.close()
    return audio_file

def transcribe_audio_locally(audio_file_path, retries=2):
    for attempt in range(retries):
        try:
            model = whisper.load_model("small")  # upgraded from tiny
            result = model.transcribe(audio_file_path)
            return result['text']
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise e

# -----------------------------
# AGENDA RESOLUTION
# -----------------------------
def analyze_agenda_resolution(agenda_points, transcript):
    resolved_keywords = ["resolved", "completed", "closed", "fixed"]
    unresolved_keywords = ["unresolved", "pending", "open", "incomplete"]

    resolved, unresolved = [], []

    for point in agenda_points:
        matches = [line for line in transcript.split(".") if point.lower() in line.lower()]
        if matches:
            if any(kw in " ".join(matches).lower() for kw in resolved_keywords):
                resolved.append(point)
            elif any(kw in " ".join(matches).lower() for kw in unresolved_keywords):
                unresolved.append(point)
            else:
                unresolved.append(point)
        else:
            unresolved.append(point)

    return resolved, unresolved

# -----------------------------
# SUMMARY GENERATION
# -----------------------------
def generate_post_meeting_summary_rag(transcript, query, vectorstore):
    llm = LangchainOpenAI(temperature=0.5, max_tokens=300)
    relevant_docs = vectorstore.similarity_search(query, k=3)
    relevant_text = "\n".join([doc.page_content for doc in relevant_docs])

    prompt_template = """
    Summarize the meeting with:
    1. Key Topics
    2. Decisions
    3. Action Items & Owners

    Context:
    {relevant_text}

    Transcript:
    {transcript}

    Query:
    {query}
    """

    truncated_transcript = truncate_text_tokens(transcript, 1500)
    truncated_relevant_text = truncate_text_tokens(relevant_text, 1500)

    prompt = PromptTemplate(
        input_variables=["relevant_text", "transcript", "query"],
        template=prompt_template
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    return llm_chain.run(
        relevant_text=truncated_relevant_text,
        transcript=truncated_transcript,
        query=query
    )

# -----------------------------
# STREAMLIT APP
# -----------------------------
def pre_meeting_document_management():
    st.header("📄 Pre-Meeting Document Management")
    uploaded_file = st.file_uploader("Upload doc (PDF, DOCX, Image)", type=["pdf", "doc", "docx", "png", "jpg", "jpeg"])
    discussion_points = st.text_area("Enter discussion points (comma separated)")
    meeting_title = st.text_input("Enter Meeting Title")
    meeting_date = st.date_input("Meeting Date")

    if uploaded_file and discussion_points and meeting_title and meeting_date:
        if st.button("Save Discussion Points"):
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                text = extract_text_from_doc(uploaded_file)
            elif uploaded_file.type.startswith("image/"):
                text = extract_text_from_image(uploaded_file)
            else:
                st.error("Unsupported file format")
                return

            st.session_state['discussion_points'] = discussion_points.split(',')
            st.session_state['vectorstore'] = create_vector_db(text)

            # Save to DB
            conn = get_db_connection()
            with conn.cursor() as cur:
                # Insert meeting
                cur.execute("INSERT INTO meetings (meeting_title, meeting_date) VALUES (%s, %s)", 
                            (meeting_title, meeting_date))
                meeting_id = cur.lastrowid

                # Insert document
                cur.execute("INSERT INTO documents (doc_name, doc_type, doc_text) VALUES (%s, %s, %s)", 
                            (uploaded_file.name, uploaded_file.type, text))

                conn.commit()
            conn.close()

            st.session_state['meeting_id'] = meeting_id
            st.success("Meeting, discussion points, and document saved!")

def agenda_creation():
    st.header("📋 Agenda Creation")
    if 'discussion_points' in st.session_state and 'vectorstore' in st.session_state:
        if st.button("Generate Agenda"):
            progress = st.progress(0)
            agenda = generate_agenda_with_discussion_points_rag(
                st.session_state['discussion_points'], st.session_state['vectorstore']
            )
            progress.progress(100)
            st.subheader("Generated Agenda")
            st.write(agenda)

            st.session_state['agenda'] = st.session_state['discussion_points']

            # Save agenda to DB
            if 'meeting_id' in st.session_state:
                conn = get_db_connection()
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO agendas (meeting_id, agenda_text) VALUES (%s, %s)", 
                                (st.session_state['meeting_id'], agenda))
                    conn.commit()
                conn.close()
    else:
        st.warning("Please upload docs and discussion points first.")

def meeting_tracking_with_agenda(video_file, agenda_points):
    st.header("🎥 Meeting Tracking & Agenda Analysis")
    progress = st.progress(0)
    audio_file = extract_audio_from_video(video_file)
    progress.progress(30)
    transcript = transcribe_audio_locally(audio_file)
    progress.progress(70)

    resolved_agenda, unresolved_agenda = analyze_agenda_resolution(agenda_points, transcript)
    progress.progress(100)

    st.subheader("Resolved Agenda Points")
    st.write(resolved_agenda if resolved_agenda else "None resolved.")

    st.subheader("Unresolved Agenda Points")
    st.write(unresolved_agenda if unresolved_agenda else "None unresolved.")

    # Save transcript to DB
    if 'meeting_id' in st.session_state:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("INSERT INTO transcripts (meeting_id, transcript) VALUES (%s, %s)", 
                        (st.session_state['meeting_id'], transcript))
            conn.commit()
        conn.close()

    os.remove(audio_file)

def post_meeting_summary():
    st.header("📝 Post-Meeting Summary")
    video_file = st.file_uploader("Upload meeting video", type=["mp4", "avi", "mov"])
    query = st.text_input("Enter query", "Key topics, decisions, action items")

    if video_file and query and 'vectorstore' in st.session_state:
        if st.button("Generate Summary"):
            progress = st.progress(0)
            audio_file = extract_audio_from_video(video_file)
            progress.progress(20)
            transcript = transcribe_audio_locally(audio_file)
            progress.progress(60)
            response = generate_post_meeting_summary_rag(transcript, query, st.session_state['vectorstore'])
            progress.progress(100)

            st.subheader("Generated Summary")
            st.write(response)

            # Save summary to DB
            if 'meeting_id' in st.session_state:
                conn = get_db_connection()
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO summaries (meeting_id, summary_text) VALUES (%s, %s)", 
                                (st.session_state['meeting_id'], response))
                    conn.commit()
                conn.close()

            os.remove(audio_file)
    else:
        st.warning("Upload video, enter query, and complete Pre-Meeting first.")

# -----------------------------
# MAIN
# -----------------------------
def main():
    st.title("🤖 Meeting Management System 🤖")
    page = st.sidebar.selectbox("Navigation", ["Pre-Meeting", "Agenda Creation", "Meeting Tracking", "Post-Meeting Summary"])

    if page == "Pre-Meeting":
        pre_meeting_document_management()
    elif page == "Agenda Creation":
        agenda_creation()
    elif page == "Meeting Tracking":
        video_file = st.file_uploader("Upload meeting video", type=["mp4", "avi", "mov"])
        if video_file and 'agenda' in st.session_state:
            st.subheader("Agenda Points:")
            st.write("\n".join(st.session_state['agenda']))
            if st.button("Track Meeting"):
                meeting_tracking_with_agenda(video_file, st.session_state['agenda'])
        else:
            st.error("Upload video and set agenda first.")
    elif page == "Post-Meeting Summary":
        post_meeting_summary()

if __name__ == "__main__":
    main()
