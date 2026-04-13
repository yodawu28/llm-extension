"""
Chunking Service - Split documents into smaller chunks for embedding
Uses LangChain's RecursiveCharacterTextSplitter with markdown-specific separators
"""

from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

from schemas.requests import PageContent


class ChunkMetadata:
    """Metadata for a single chunk"""

    def __init__(
        self,
        page_title: str,
        page_url: str,
        source_type: str,
        chunk_index: int,
        total_chunks: int,
        char_start: int,
        char_end: int
    ):
        self.page_title = page_title
        self.page_url = page_url
        self.source_type = source_type
        self.chunk_index = chunk_index
        self.total_chunks = total_chunks
        self.char_start = char_start
        self.char_end = char_end

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_title": self.page_title,
            "page_url": self.page_url,
            "source_type": self.source_type,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "char_start": self.char_start,
            "char_end": self.char_end
        }


class Chunk:
    """A single text chunk with metadata"""

    def __init__(self, text: str, metadata: ChunkMetadata):
        self.text = text
        self.metadata = metadata

    def __repr__(self):
        return f"Chunk(text={self.text[:50]}..., page={self.metadata.page_title})"


class ChunkingService:
    """Split documents into chunks optimized for RAG"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize chunking service

        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Use markdown-specific separators for better semantic chunking
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",  # Multiple newlines (section breaks)
                "\n\n",    # Double newlines (paragraphs)
                "\n",      # Single newlines
                ". ",      # Sentences
                ", ",      # Clauses
                " ",       # Words
                ""         # Characters
            ]
        )

    def chunk_page(self, page: PageContent) -> List[Chunk]:
        """
        Split a single page into chunks

        Args:
            page: Page content with title, url, markdown

        Returns:
            List of Chunk objects with metadata
        """
        if not page.markdown or not page.markdown.strip():
            return []

        # Split markdown into chunks
        text_chunks = self.splitter.split_text(page.markdown)

        # Create Chunk objects with metadata
        chunks = []
        char_position = 0

        for i, text in enumerate(text_chunks):
            metadata = ChunkMetadata(
                page_title=page.title,
                page_url=page.url,
                source_type=page.source_type,
                chunk_index=i,
                total_chunks=len(text_chunks),
                char_start=char_position,
                char_end=char_position + len(text)
            )

            chunks.append(Chunk(text=text, metadata=metadata))

            # Update position for next chunk
            # Account for overlap by moving back slightly
            char_position += len(text) - self.chunk_overlap

        return chunks

    def chunk_pages(self, pages: List[PageContent]) -> List[Chunk]:
        """
        Split multiple pages into chunks

        Args:
            pages: List of page contents

        Returns:
            Flat list of all chunks from all pages
        """
        all_chunks = []

        for page in pages:
            page_chunks = self.chunk_page(page)
            all_chunks.extend(page_chunks)

        return all_chunks

    def get_chunk_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Get statistics about chunks

        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_chars": 0,
                "avg_chunk_size": 0,
                "pages_count": 0
            }

        total_chars = sum(len(chunk.text) for chunk in chunks)
        unique_pages = set(chunk.metadata.page_url for chunk in chunks)

        return {
            "total_chunks": len(chunks),
            "total_chars": total_chars,
            "avg_chunk_size": total_chars // len(chunks),
            "pages_count": len(unique_pages),
            "chunks_per_page": len(chunks) / len(unique_pages) if unique_pages else 0
        }
