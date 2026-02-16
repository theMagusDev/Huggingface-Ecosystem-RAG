import os
import shutil
import glob
from git import Repo
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from tqdm import tqdm
import config
import stat

def remove_readonly(func, path, excinfo):
    """A helper function to delete read-only files (for Windows)"""
    os.chmod(path, stat.S_IWRITE)
    func(path)

class KnowledgeBaseBuilder:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        
    def _clean_url(self, url):
        return url.strip().strip("[]()")

    def download_repos(self):
        if not os.path.exists(config.DOCS_TEMP_DIR):
            os.makedirs(config.DOCS_TEMP_DIR)
            
        docs = []
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN, chunk_size=1000, chunk_overlap=200
        )

        for lib_name, (url, doc_path) in config.LIBS_CONFIG.items():
            print(f"📥 Processing {lib_name}...")
            repo_path = os.path.join(config.DOCS_TEMP_DIR, lib_name)
            clean_url = self._clean_url(url)
            
            # Clone if not exists
            if not os.path.exists(repo_path):
                Repo.clone_from(clean_url, repo_path, depth=1)
            
            # Read files
            full_doc_path = os.path.join(repo_path, doc_path)
            md_files = glob.glob(os.path.join(full_doc_path, "**/*.md"), recursive=True)
            
            lib_docs = []
            for file_path in tqdm(md_files, desc=f"Loading {lib_name}"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    if len(content) < 50: continue
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": os.path.relpath(file_path, full_doc_path),
                            "lib_name": lib_name
                        }
                    )
                    lib_docs.append(doc)
                except Exception:
                    continue
            
            chunks = splitter.split_documents(lib_docs)
            docs.extend(chunks)
            print(f"   ✂️  {len(chunks)} chunks created.")
            
        return docs

    def build_index(self):
        print("🚀 Starting ingestion pipeline...")
        docs = self.download_repos()
        
        if os.path.exists(config.DB_PATH):
            shutil.rmtree(config.DB_PATH)
            
        print(f"💾 Creating Vector DB with {len(docs)} chunks...")
        Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=config.DB_PATH
        )
        print("✅ Index created successfully!")
        # Cleanup temp files
        if os.path.exists(config.DOCS_TEMP_DIR):
            print(f"🧹 Cleaning up {config.DOCS_TEMP_DIR}...")
            try:
                shutil.rmtree(config.DOCS_TEMP_DIR, onexc=remove_readonly)
                print("✨ Cleanup finished.")
            except Exception as e:
                print(f"⚠️ Could not delete temp docs: {e}. You can delete them manually.")

if __name__ == "__main__":
    builder = KnowledgeBaseBuilder()
    builder.build_index()
