import streamlit as st
import os
import whisper
import numpy as np
from moviepy.editor import VideoFileClip
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI as LangchainOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import docx
import tiktoken

# Set OpenAI API key 
os.environ['OPENAI_API_KEY'] = "sk-proj-n7I2ffVOFuT4pAJ1ldvBT3BlbkFJmgAqg9DWXxO4NRvFsgUj"

# This function to extract text from PDF files
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# This function to extract text from DOC/DOCX files
def extract_text_from_doc(doc_file):
    doc = docx.Document(doc_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# This function to extract text from image files using OCR
def extract_text_from_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

# Thus function to count the number of tokens in a text
def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

# This function to truncate text to a maximum number of tokens
def truncate_text_tokens(text, max_tokens):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        return encoding.decode(tokens[:max_tokens])
    return text

# This function to create a vector database from text
def create_vector_db(text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts, embeddings)

# Generate an agenda based on discussion points and relevant documents
def generate_agenda_with_discussion_points_rag(discussion_points, vectorstore):
    llm = LangchainOpenAI(temperature=0.5, max_tokens=400)
    relevant_docs = vectorstore.similarity_search(" ".join(discussion_points), k=5)
    relevant_text = "\n".join([doc.page_content for doc in relevant_docs])

    prompt_template = """
    You are a helpful assistant that organizes meeting agendas.
    The following are the discussion points provided by the participants:

    {discussion_points}

    Based on the document and relevant information retrieved, please organize the points into a coherent agenda:

    {relevant_text}

    Organize the points and link related topics where possible to streamline the meeting flow.
    """

    max_input_tokens = 3000
    truncated_relevant_text = truncate_text_tokens(relevant_text, max_input_tokens)

    prompt = PromptTemplate(
        input_variables=["discussion_points", "relevant_text"],
        template=prompt_template
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    return llm_chain.run(discussion_points=discussion_points, relevant_text=truncated_relevant_text)

# Extract audio from a video file
def extract_audio_from_video(video_file):
    video_clip = VideoFileClip(video_file.name)
    audio_file = "temp_audio.wav"
    video_clip.audio.write_audiofile(audio_file)
    video_clip.close()
    return audio_file

# Transcribe audio file to text using Whisper model
def transcribe_audio_locally(audio_file_path):
    model = whisper.load_model("tiny")
    result = model.transcribe(audio_file_path)
    return result['text']

# Analyze agenda resolution based on the meeting transcript
def analyze_agenda_resolution(agenda_points, transcript):
    transcript_lower = transcript.lower()
    resolved_keywords = ["resolved", "completed", "closed", "fixed"] # resolved keywords
    unresolved_keywords = ["unresolved", "pending", "needs further discussion", "open", "incomplete"] #unresolved code 
    resolved_agenda = []
    unresolved_agenda = []

    for point in agenda_points:
        point_lower = point.lower()
        if point_lower in transcript_lower:
            if any(keyword in transcript_lower for keyword in resolved_keywords):
                resolved_agenda.append(point)
            elif any(keyword in transcript_lower for keyword in unresolved_keywords):
                unresolved_agenda.append(point)
        else:
            unresolved_agenda.append(point)

    return resolved_agenda, unresolved_agenda

# Generate a post-meeting summary based on transcript and query
def generate_post_meeting_summary_rag(transcript, query, vectorstore):
    llm = LangchainOpenAI(temperature=0.5, max_tokens=300)
    relevant_docs = vectorstore.similarity_search(query, k=3)
    relevant_text = "\n".join([doc.page_content for doc in relevant_docs])

    prompt_template = f"""
    Based on the following relevant transcript segments and context, provide a summary that includes:
    1. Key Discussion Topics: Main topics discussed.
    2. Decisions Made: Decisions made during the meeting.
    3. Action Items and Owners: Tasks assigned, who is responsible.

    Relevant Context:
    {{relevant_text}}

    Relevant Transcript:
    {{transcript}}

    Query:
    {{query}}
    """

    max_input_tokens = 3000
    truncated_transcript = truncate_text_tokens(transcript, max_input_tokens // 2)
    truncated_relevant_text = truncate_text_tokens(relevant_text, max_input_tokens // 2)

    prompt = PromptTemplate(
        input_variables=["relevant_text", "transcript", "query"],
        template=prompt_template
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    return llm_chain.run(relevant_text=truncated_relevant_text, transcript=truncated_transcript, query=query)

# Streamlit  code 

# Function to manage pre-meeting documents and discussion points
def pre_meeting_document_management():
    st.header("📄 Pre-Meeting Document Management")

    uploaded_file = st.file_uploader("Upload a document (PDF, DOC, DOCX, Image)", type=["pdf", "doc", "docx", "png", "jpg", "jpeg"])
    discussion_points = st.text_area("Enter discussion points (separated by commas)")

    if uploaded_file and discussion_points:
        if st.button("Save Discussion Points"):
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(uploaded_file)
                # this  identify files created using the Open XML file formats specifically for Microsoft Word 
            elif uploaded_file.type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                text = extract_text_from_doc(uploaded_file)
            elif uploaded_file.type.startswith("image/"):
                text = extract_text_from_image(uploaded_file)
            else:
                st.error("Unsupported file format")
                return

            st.session_state['discussion_points'] = discussion_points.split(',')
            st.session_state['vectorstore'] = create_vector_db(text)
            st.success("Discussion points and document saved!")

# Function to create and display the meeting agenda
def agenda_creation():
    st.header("📋 Agenda Creation")

    if 'discussion_points' in st.session_state and 'vectorstore' in st.session_state:
        if st.button("Generate Agenda"):
            with st.spinner("Generating agenda..."):
                agenda = generate_agenda_with_discussion_points_rag(st.session_state['discussion_points'], st.session_state['vectorstore'])
                st.subheader("Generated Agenda")
                st.write(agenda)
                st.session_state['agenda'] = st.session_state['discussion_points']
    else:
        st.warning("Please complete Pre-Meeting Document Management first.")

# Function to track meeting video and analyze agenda resolution
def meeting_tracking_with_agenda(video_file, agenda_points):
    st.header("🎥 Meeting Tracking and Agenda Analysis")
    with st.spinner("Processing the meeting video and analyzing agenda... 🚀"):
        audio_file = extract_audio_from_video(video_file)
        transcript = transcribe_audio_locally(audio_file)

        resolved_agenda, unresolved_agenda = analyze_agenda_resolution(agenda_points, transcript)

        st.subheader("Resolved Agenda Points")
        st.write("\n".join(resolved_agenda) if resolved_agenda else "No points resolved.")

        st.subheader("Unresolved Agenda Points")
        st.write("\n".join(unresolved_agenda) if unresolved_agenda else "No points unresolved.")

        os.remove(audio_file)

# Function to generate post-meeting summary based on video and query
def post_meeting_summary():
    st.header("📝 Post-Meeting Summary")

    video_file = st.file_uploader("Upload a meeting video", type=["mp4", "avi", "mov"])
    query = st.text_input("Enter specific query or key points to summarize", "Key discussion topics, decisions made, action items")

    if video_file and query and 'vectorstore' in st.session_state:
        if st.button("Generate Post-Meeting Summary"):
            with st.spinner("Generating summary... 🚀"):
                audio_file = extract_audio_from_video(video_file)
                transcript = transcribe_audio_locally(audio_file)

                response = generate_post_meeting_summary_rag(transcript, query, st.session_state['vectorstore'])
                st.subheader("Generated Post-Meeting Summary")
                st.write(response)

                os.remove(audio_file)
    else:
        st.warning("Please upload a video, enter a query, and ensure document management is complete.")

# Main Streamlit app function
def main():
    st.title("🤖 Meeting Management System 🤖")

    page = st.sidebar.selectbox("Navigation", ["Pre-Meeting", "Agenda Creation", "Meeting Tracking", "Post-Meeting Summary"])

    if page == "Pre-Meeting":
        pre_meeting_document_management()
    elif page == "Agenda Creation":
        agenda_creation()
    elif page == "Meeting Tracking":
        video_file = st.file_uploader("Upload meeting video", type=["mp4", "avi", "mov"])
        if video_file:
            if 'agenda' in st.session_state:
                st.subheader("Tracking based on the following agenda:")
                st.write("\n".join(st.session_state['agenda']))

                if st.button("Track Meeting and Analyze Agenda"):
                    meeting_tracking_with_agenda(video_file, st.session_state['agenda'])
            else:
                st.error("No agenda points available. Please upload documents and enter discussion points first.")
    elif page == "Post-Meeting Summary":
        post_meeting_summary()

# Run the Streamlit app
if __name__ == "__main__":
    main()
