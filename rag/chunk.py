import json
from pathlib import Path


def chunk_pages(pages, size=250, overlap=40):
    """
    Splits each page into overlapping chunks of words.

    Parameters:
        pages (list): A list of page dicts, as returned by rag.load.load_pages. Each must have:
            - doc_id: The document ID (string)
            - page: The page number (int)
            - text: The text content of the page (string)

        size (int): Maximum number of words per chunk.
        overlap (int): Number of words shared by consecutive chunks on the same page.

    Returns:
        list: A list of chunk records with the following keys:
            - chunk_id: Unique identifier for the chunk, e.g. "p463-p34-c2".
            - doc_id: The document ID.
            - page: The page number.
            - text: The content of the chunk.
    """
    if not 0 <= overlap < size:
        raise ValueError(f"need 0 <= overlap < size, got size={size}, overlap={overlap}")

    chunks = []

    for page in pages:
        doc_id = page['doc_id']
        page_num = page['page']
        text = page['text']

        # Split into words on whitespace
        words = text.split()

        start = 0
        chunk_number = 1

        while start < len(words):
            end = min(start + size, len(words))
            chunk_text = ' '.join(words[start:end])
            chunk_id = f"{doc_id}-p{page_num}-c{chunk_number}"

            chunks.append({
                'chunk_id': chunk_id,
                'doc_id': doc_id,
                'page': page_num,
                'text': chunk_text
            })

            # Stop once a chunk reaches the end of the page; another step would
            # only produce a chunk made entirely of words already in this one.
            if end == len(words):
                break

            start += size - overlap
            chunk_number += 1

    return chunks


def save_chunks(chunks, path):
    """Writes chunks to a JSONL file, one chunk per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')


def load_chunks(path):
    """Reads chunks written by save_chunks, in the same order."""
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]
