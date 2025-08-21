from src.ingest import chunk_text

def test_chunking():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    # Expected starts: 0, 8, 16, 24
    assert chunks[0] == "abcdefghij"
    assert chunks[1].startswith("ijklmnop")
    assert len(chunks) >= 3