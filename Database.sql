-- Create database (if not exists)
CREATE DATABASE IF NOT EXISTS meetingdb;
USE meetingdb;

-- Table: documents
CREATE TABLE documents (
    doc_id INT AUTO_INCREMENT PRIMARY KEY,
    doc_name VARCHAR(255) NOT NULL,
    doc_type VARCHAR(50),             -- pdf, docx, image
    doc_text LONGTEXT NOT NULL,       -- extracted text (OCR or plain)
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table: meetings
CREATE TABLE meetings (
    meeting_id INT AUTO_INCREMENT PRIMARY KEY,
    meeting_title VARCHAR(255) NOT NULL,
    meeting_date DATETIME NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table: agendas
CREATE TABLE agendas (
    agenda_id INT AUTO_INCREMENT PRIMARY KEY,
    meeting_id INT,
    agenda_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (meeting_id) REFERENCES meetings(meeting_id) ON DELETE CASCADE
);

-- Table: transcripts
CREATE TABLE transcripts (
    transcript_id INT AUTO_INCREMENT PRIMARY KEY,
    meeting_id INT,
    transcript LONGTEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (meeting_id) REFERENCES meetings(meeting_id) ON DELETE CASCADE
);

-- Table: summaries
CREATE TABLE summaries (
    summary_id INT AUTO_INCREMENT PRIMARY KEY,
    meeting_id INT,
    summary_text LONGTEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (meeting_id) REFERENCES meetings(meeting_id) ON DELETE CASCADE
);

-- Table: embeddings (optional for RAG persistence)
CREATE TABLE embeddings (
    embedding_id INT AUTO_INCREMENT PRIMARY KEY,
    doc_id INT,
    chunk_text TEXT NOT NULL,
    embedding BLOB NOT NULL,   -- store vector as binary or JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);
