from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

text_dir = Path("roag_texts")
docs = []

for txt_file in text_dir.glob("*.txt"):
    with open(txt_file, "r", encoding="utf-8") as f:
        content = f.read()
    # skip short/empty content
    if len(content) < 100:
        continue

    docs.append(Document(
        page_content=content,
        metadata={"source": txt_file.name}
    ))

# split params
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = splitter.split_documents(docs)
print(f"Всего чанков: {len(chunks)}")