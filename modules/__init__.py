# __init__.py for modules package
# This file allows you to import all modules as a package

from .ocr_module import extract_text_from_pdf, extract_text_from_doc, extract_text_from_image
from .transcription import transcribe_audio_locally, extract_audio_from_video
from .summarizer import generate_agenda_with_discussion_points_rag, generate_post_meeting_summary_rag
from .db_connection import get_db_connection, create_tables, test_crud_operations

__all__ = [
    "extract_text_from_pdf",
    "extract_text_from_doc",
    "extract_text_from_image",
    "transcribe_audio_locally",
    "extract_audio_from_video",
    "generate_agenda_with_discussion_points_rag",
    "generate_post_meeting_summary_rag",
    "get_db_connection",
    "create_tables",
    "test_crud_operations",
]
