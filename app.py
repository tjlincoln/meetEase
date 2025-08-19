# app.py
import streamlit as st
from modules import (
    extract_text_from_pdf,
    transcribe_audio_locally,
    generate_post_meeting_summary_rag
)

def main():
    st.title("🤖 MeetEase - Meeting Management Tool 🤖")
    st.write("Upload documents, transcribe meetings, and generate summaries.")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        st.subheader("Extracted Text")
        st.write(text)

    audio_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3"])
    if audio_file:
        transcript = transcribe_audio_locally(audio_file.name)
        st.subheader("Transcript")
        st.write(transcript)

        query = st.text_input("Enter query for summary", "Key discussion points")
        if st.button("Generate Summary"):
            # For demo: no vector DB passed yet, you can add it later
            response = generate_post_meeting_summary_rag(transcript, query, None)
            st.subheader("Summary")
            st.write(response)

if __name__ == "__main__":
    main()
