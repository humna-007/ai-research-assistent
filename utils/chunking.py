"""Document chunking using LangChain's RecursiveCharacterTextSplitter."""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents: list[dict], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in documents:
        for i, piece in enumerate(splitter.split_text(doc["text"])):
            chunks.append({"text": piece, "metadata": {**doc["metadata"], "chunk_index": i}})
    return chunks