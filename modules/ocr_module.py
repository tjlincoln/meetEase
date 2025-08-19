from PIL import Image
import pytesseract
import fitz
import docx

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "".join([page.get_text() for page in doc])

def extract_text_from_doc(doc_file):
    doc = docx.Document(doc_file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_image(image_file):
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)
