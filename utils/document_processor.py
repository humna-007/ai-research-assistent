"""
Document extraction for PDF, DOCX, and TXT files.
Extracts text while preserving metadata needed for source citations.
"""

import fitz  # PyMuPDF
import docx
from io import BytesIO

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
MAX_FILE_SIZE_MB = 25


class DocumentProcessingError(Exception):
    """Raised when a document cannot be parsed or is invalid."""
    pass


def _get_extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


def validate_file(uploaded_file) -> None:
    ext = _get_extension(uploaded_file.name)
    if ext not in ALLOWED_EXTENSIONS:
        raise DocumentProcessingError(
            f"'{uploaded_file.name}' has an unsupported format. Allowed: PDF, DOCX, TXT."
        )
    if uploaded_file.size == 0:
        raise DocumentProcessingError(f"'{uploaded_file.name}' is empty.")
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise DocumentProcessingError(
            f"'{uploaded_file.name}' is {size_mb:.1f}MB, over the {MAX_FILE_SIZE_MB}MB limit."
        )


def extract_pdf(uploaded_file) -> list[dict]:
    pages = []
    try:
        file_bytes = uploaded_file.read()
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            if doc.page_count == 0:
                raise DocumentProcessingError(f"'{uploaded_file.name}' has no pages.")
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append({
                        "text": text,
                        "metadata": {"source": uploaded_file.name, "page": page_num, "type": "pdf"},
                    })
    except DocumentProcessingError:
        raise
    except Exception as exc:
        raise DocumentProcessingError(
            f"Could not read '{uploaded_file.name}'. It may be corrupted or password-protected."
        ) from exc

    if not pages:
        raise DocumentProcessingError(
            f"No extractable text in '{uploaded_file.name}' (possibly a scanned/image-only PDF)."
        )
    return pages


def extract_docx(uploaded_file) -> list[dict]:
    try:
        file_bytes = BytesIO(uploaded_file.read())
        document = docx.Document(file_bytes)
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs).strip()
    except Exception as exc:
        raise DocumentProcessingError(f"Could not read '{uploaded_file.name}'. It may be corrupted.") from exc

    if not text:
        raise DocumentProcessingError(f"'{uploaded_file.name}' has no extractable text.")

    return [{"text": text, "metadata": {"source": uploaded_file.name, "page": None, "type": "docx"}}]


def extract_txt(uploaded_file) -> list[dict]:
    try:
        raw = uploaded_file.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        text = text.strip()
    except Exception as exc:
        raise DocumentProcessingError(f"Could not read '{uploaded_file.name}' as text.") from exc

    if not text:
        raise DocumentProcessingError(f"'{uploaded_file.name}' is empty.")

    return [{"text": text, "metadata": {"source": uploaded_file.name, "page": None, "type": "txt"}}]


def process_uploaded_file(uploaded_file) -> list[dict]:
    validate_file(uploaded_file)
    ext = _get_extension(uploaded_file.name)
    if ext == "pdf":
        return extract_pdf(uploaded_file)
    if ext == "docx":
        return extract_docx(uploaded_file)
    return extract_txt(uploaded_file)


def process_uploaded_files(uploaded_files) -> tuple[list[dict], list[str]]:
    """One bad file shouldn't block the rest of the batch."""
    documents, errors = [], []
    for f in uploaded_files:
        try:
            documents.extend(process_uploaded_file(f))
        except DocumentProcessingError as e:
            errors.append(str(e))
    return documents, errors