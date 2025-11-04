"""
RAG Build Module - Document Ingestion and Index Building.
Processes PDF/TXT documents, creates chunks, generates embeddings, and builds FAISS indexes.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from .config import get_config
from .utils import get_logger, list_files, chunk_text, ProgressTracker


logger = get_logger(__name__)


class Document:
    """Represents a document with metadata."""
    
    def __init__(self, text: str, source: str, page: Optional[int] = None, subject: Optional[str] = None):
        """
        Initialize document.
        
        Args:
            text: Document text content
            source: Source file name or identifier
            page: Page number (for PDFs)
            subject: Subject category (maths, physique, anglais)
        """
        self.text = text
        self.source = source
        self.page = page
        self.subject = subject
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "subject": self.subject
        }


class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    
    def __init__(self, text: str, doc_source: str, page: Optional[int] = None, 
                 chunk_idx: int = 0, subject: Optional[str] = None):
        """
        Initialize document chunk.
        
        Args:
            text: Chunk text content
            doc_source: Source document file name
            page: Page number
            chunk_idx: Index of this chunk within the document
            subject: Subject category
        """
        self.text = text
        self.doc_source = doc_source
        self.page = page
        self.chunk_idx = chunk_idx
        self.subject = subject
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "source": self.doc_source,
            "page": self.page,
            "chunk_idx": self.chunk_idx,
            "subject": self.subject
        }


class RAGBuilder:
    """Builds RAG indexes from documents."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize RAG builder.
        
        Args:
            config_path: Path to config file
        """
        self.config = get_config(config_path)
        self.embedding_model = None
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load the sentence transformer embedding model."""
        logger.info(f"Loading embedding model: {self.config.rag.embedding_model}")
        self.embedding_model = SentenceTransformer(
            self.config.rag.embedding_model,
            device=self.config.rag.embedding_device
        )
        logger.info("Embedding model loaded successfully")
    
    def load_text_file(self, file_path: Path) -> Document:
        """
        Load a text file.
        
        Args:
            file_path: Path to text file
        
        Returns:
            Document object
        """
        logger.debug(f"Loading text file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return Document(
            text=text,
            source=file_path.name,
            page=None,
            subject=file_path.parent.name
        )
    
    def load_pdf_file(self, file_path: Path) -> List[Document]:
        """
        Load a PDF file using PyMuPDF.
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            List of Document objects (one per page)
        """
        logger.debug(f"Loading PDF file: {file_path}")
        
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.error("PyMuPDF not installed. Install with: pip install pymupdf")
            raise
        
        documents = []
        pdf_doc = fitz.open(file_path)
        
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            text = page.get_text()
            
            if text.strip():  # Only add non-empty pages
                documents.append(Document(
                    text=text,
                    source=file_path.name,
                    page=page_num + 1,
                    subject=file_path.parent.name
                ))
        
        pdf_doc.close()
        logger.debug(f"Loaded {len(documents)} pages from {file_path.name}")
        return documents
    
    def load_documents_from_directory(self, directory: Path) -> List[Document]:
        """
        Load all documents from a directory.
        
        Args:
            directory: Directory containing documents
        
        Returns:
            List of Document objects
        """
        logger.info(f"Loading documents from: {directory}")
        
        documents = []
        
        # Load text files
        text_files = list_files(directory, ['.txt'])
        for file_path in text_files:
            try:
                doc = self.load_text_file(file_path)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        # Load PDF files
        pdf_files = list_files(directory, ['.pdf'])
        for file_path in pdf_files:
            try:
                docs = self.load_pdf_file(file_path)
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of Document objects
        
        Returns:
            List of DocumentChunk objects
        """
        logger.info("Chunking documents...")
        
        chunks = []
        progress = ProgressTracker(len(documents), "Chunking documents")
        
        for doc in documents:
            text_chunks = chunk_text(
                doc.text,
                chunk_size=self.config.rag.chunk_size,
                overlap=self.config.rag.chunk_overlap
            )
            
            for idx, text_chunk in enumerate(text_chunks):
                chunks.append(DocumentChunk(
                    text=text_chunk,
                    doc_source=doc.source,
                    page=doc.page,
                    chunk_idx=idx,
                    subject=doc.subject
                ))
            
            progress.update()
        
        progress.finish()
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def create_embeddings(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """
        Create embeddings for all chunks.
        
        Args:
            chunks: List of DocumentChunk objects
        
        Returns:
            Numpy array of embeddings
        """
        logger.info("Creating embeddings...")
        
        texts = [chunk.text for chunk in chunks]
        
        # Encode in batches for efficiency
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32
        )
        
        logger.info(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
        
        Returns:
            FAISS index
        """
        logger.info("Building FAISS index...")
        
        dimension = embeddings.shape[1]
        
        # Use IndexFlatL2 for exact search (good for small to medium datasets)
        # For larger datasets, consider IndexIVFFlat or IndexHNSWFlat
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        logger.info(f"FAISS index built with {index.ntotal} vectors")
        return index
    
    def save_index(self, index: faiss.Index, chunks: List[DocumentChunk], 
                   output_path: Path):
        """
        Save FAISS index and chunk metadata.
        
        Args:
            index: FAISS index
            chunks: List of DocumentChunk objects
            output_path: Path to save index file
        """
        logger.info(f"Saving index to: {output_path}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(index, str(output_path))
        
        # Save chunk metadata
        metadata_path = output_path.with_suffix('.pkl')
        chunk_dicts = [chunk.to_dict() for chunk in chunks]
        with open(metadata_path, 'wb') as f:
            pickle.dump(chunk_dicts, f)
        
        logger.info(f"Index and metadata saved successfully")
    
    def build_index_for_subject(self, subject: str) -> bool:
        """
        Build FAISS index for a specific subject.
        
        Args:
            subject: Subject name (maths, physique, anglais)
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Building index for subject: {subject}")
        
        try:
            # Load documents
            subject_dir = self.config.data_dir / subject
            if not subject_dir.exists():
                logger.warning(f"Subject directory not found: {subject_dir}")
                return False
            
            documents = self.load_documents_from_directory(subject_dir)
            
            if not documents:
                logger.warning(f"No documents found for subject: {subject}")
                return False
            
            # Chunk documents
            chunks = self.chunk_documents(documents)
            
            # Create embeddings
            embeddings = self.create_embeddings(chunks)
            
            # Build FAISS index
            index = self.build_faiss_index(embeddings)
            
            # Save index
            output_path = self.config.get_index_path(subject)
            self.save_index(index, chunks, output_path)
            
            logger.info(f"Successfully built index for {subject}")
            return True
        
        except Exception as e:
            logger.error(f"Error building index for {subject}: {e}")
            return False
    
    def build_all_indexes(self) -> Dict[str, bool]:
        """
        Build FAISS indexes for all subjects.
        
        Returns:
            Dictionary mapping subject names to success status
        """
        logger.info("Building indexes for all subjects")
        
        subjects = ["maths", "physique", "anglais"]
        results = {}
        
        for subject in subjects:
            results[subject] = self.build_index_for_subject(subject)
        
        # Summary
        successful = sum(1 for v in results.values() if v)
        logger.info(f"Index building complete: {successful}/{len(subjects)} successful")
        
        return results


def main():
    """Main entry point for building indexes."""
    from .utils import setup_logging
    
    # Setup logging
    config = get_config()
    setup_logging(
        log_level=config.orchestrator.log_level,
        log_file=str(config.logs_dir / "rag_build.log")
    )
    
    # Build indexes
    builder = RAGBuilder()
    results = builder.build_all_indexes()
    
    # Print results
    print("\n" + "="*50)
    print("Index Building Results")
    print("="*50)
    for subject, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{subject:12} : {status}")
    print("="*50)


if __name__ == "__main__":
    main()
