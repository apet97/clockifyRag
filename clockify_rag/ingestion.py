"""Document ingestion utilities for various file formats.

This module provides utilities to convert different document formats
(Markdown, HTML, PDF, etc.) into normalized text for the RAG pipeline.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import PyPDF2

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("PyPDF2 not available. PDF support will be limited.")

try:
    from bs4 import BeautifulSoup

    HTML_AVAILABLE = True
except ImportError:
    HTML_AVAILABLE = False
    logger.warning("BeautifulSoup4 not available. HTML support will be limited.")


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        Extracted text content from the PDF

    Raises:
        ValueError: If PDF extraction fails
    """
    if not PDF_AVAILABLE:
        raise ValueError("PyPDF2 not installed. Install with: pip install PyPDF2")

    try:
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF {file_path}: {str(e)}")


def extract_text_from_html(file_path: str) -> str:
    """Extract text from HTML file.

    Args:
        file_path: Path to the HTML file

    Returns:
        Extracted text content from the HTML

    Raises:
        ValueError: If HTML parsing fails
    """
    if not HTML_AVAILABLE:
        raise ValueError("BeautifulSoup4 not installed. Install with: pip install beautifulsoup4")

    try:
        with open(file_path, "r", encoding="utf-8") as html_file:
            content = html_file.read()
            soup = BeautifulSoup(content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text()

            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

        return text
    except Exception as e:
        raise ValueError(f"Failed to extract text from HTML {file_path}: {str(e)}")


def extract_text_from_markdown(file_path: str) -> str:
    """Extract text from Markdown file.

    Args:
        file_path: Path to the Markdown file

    Returns:
        Extracted text content from the Markdown
    """
    with open(file_path, "r", encoding="utf-8") as md_file:
        return md_file.read()


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from plain text file.

    Args:
        file_path: Path to the text file

    Returns:
        Content of the text file
    """
    with open(file_path, "r", encoding="utf-8") as txt_file:
        return txt_file.read()


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file.

    Args:
        file_path: Path to the DOCX file

    Returns:
        Extracted text content from the DOCX

    Raises:
        ValueError: If DOCX parsing fails or dependency not available
    """
    try:
        import docx  # type: ignore[import-untyped]
    except ImportError:
        raise ValueError("python-docx not installed. Install with: pip install python-docx")

    try:
        doc = docx.Document(file_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return "\n".join(text)
    except Exception as e:
        raise ValueError(f"Failed to extract text from DOCX {file_path}: {str(e)}")


def convert_to_markdown_format(content: str, title: str = "Document Content", url: str = "") -> str:
    """Convert raw text content to markdown format expected by RAG system.

    Args:
        content: Raw text content
        title: Title for the markdown article
        url: Optional URL for the markdown article

    Returns:
        Markdown formatted string with proper RAG structure
    """
    # Clean and normalize content
    content = content.strip()

    # Split into logical sections if possible
    # Look for common section indicators (headers, list items, etc.)
    lines = content.split("\n")
    processed_lines = []

    for line in lines:
        line = line.strip()
        if line:
            processed_lines.append(line)

    # Join lines back together
    cleaned_content = "\n".join(processed_lines)

    # Format as markdown with RAG expected structure
    md_content = f"""# [ARTICLE] {title}
{url if url else ''}

{cleaned_content}
"""
    return md_content


def ingest_document(file_path: str | Path, output_path: Optional[str] = None) -> str:
    """Ingest a document of various formats and convert to normalized format for RAG.

    Args:
        file_path: Path to the input document
        output_path: Optional path to save the converted markdown (if None, returns string)

    Returns:
        Converted markdown content or path to saved file
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    # Determine file type and extract text
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        text_content = extract_text_from_pdf(str(file_path))
    elif extension in [".html", ".htm"]:
        text_content = extract_text_from_html(str(file_path))
    elif extension in [".md", ".markdown"]:
        text_content = extract_text_from_markdown(str(file_path))
    elif extension == ".txt":
        text_content = extract_text_from_txt(str(file_path))
    elif extension == ".docx":
        text_content = extract_text_from_docx(str(file_path))
    else:
        # Treat as plain text by default
        logger.warning(f"Unknown file type {extension}, treating as plain text")
        text_content = extract_text_from_txt(str(file_path))

    # Convert to RAG-compatible markdown format
    title = file_path.stem
    markdown_content = convert_to_markdown_format(text_content, title=title, url=f"file://{file_path.absolute()}")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        logger.info(f"Converted document saved to: {output_path}")
        return output_path
    else:
        return markdown_content


def ingest_directory(
    directory_path: str | Path, output_path: Optional[str] = None, supported_extensions: Optional[List[str]] = None
) -> str:
    """Ingest all documents in a directory and combine into a single knowledge base.

    Args:
        directory_path: Path to the directory containing documents
        output_path: Optional path to save the combined markdown (if None, returns string)
        supported_extensions: List of file extensions to process (default: common document types)

    Returns:
        Combined markdown content or path to saved file
    """
    if supported_extensions is None:
        supported_extensions = [".pdf", ".html", ".htm", ".md", ".txt", ".docx"]

    directory_path = Path(directory_path)
    if not directory_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")

    combined_content = []

    for file_path in directory_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                logger.info(f"Processing: {file_path}")
                content = ingest_document(str(file_path))
                combined_content.append(content)
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {str(e)}")
                continue

    combined_markdown = "\n\n".join(combined_content)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(combined_markdown)
        logger.info(f"Combined knowledge base saved to: {output_path}")
        return output_path
    else:
        return combined_markdown


def validate_ingestion_output(content: str) -> Tuple[bool, List[str]]:
    """Validate that ingestion output follows RAG expected format.

    Args:
        content: Markdown content to validate

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check for ARTICLE headers
    article_headers = re.findall(r"^# \[ARTICLE\].*$", content, re.MULTILINE)
    if not article_headers:
        issues.append("No ARTICLE headers found. Format should be: # [ARTICLE] Title")

    # Check for minimum content
    if len(content.strip()) < 10:
        issues.append("Content appears to be too short")

    # Check for proper structure
    lines = content.split("\n")
    has_content_after_header = any(
        line.strip() and not line.startswith("# [ARTICLE]") and not line.startswith("http") for line in lines
    )
    if not has_content_after_header and article_headers:
        issues.append("No content found after ARTICLE header")

    return len(issues) == 0, issues


# For backward compatibility with existing code
def build_docs_from_source(source_path: str | Path, output_path: Optional[str] = None) -> str:
    """Backward-compatible function to build docs from various sources.

    Args:
        source_path: Path to source document or directory
        output_path: Optional path to save output

    Returns:
        Processed markdown content
    """
    source_path = Path(source_path)

    if source_path.is_dir():
        return ingest_directory(str(source_path), output_path)
    else:
        return ingest_document(str(source_path), output_path)
