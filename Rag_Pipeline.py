import getpass
import os
import sys
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract

# --- All Imports ---
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import torch

# Add this import (with fallback for different LangChain versions)
from langchain_core.documents import Document

# --- Main Application Logic ---

class RAGBOT:
    """A simple RAG bot class to encapsulate the RAG functionality."""
    def __init__(self, llm, embeddings, vector_store):
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store

    @staticmethod
    def initialize_models():
        """Initializes and returns the LLM and embedding models."""
        print("🧠 Initializing models...")
        
        # Load environment variables
        load_dotenv()
        google_api_key = os.environ.get("GEMINI_API_KEY")
        huggingface_api_key = os.environ.get("HUGGINGFACE_API_TOKEN")

        if not google_api_key:
            google_api_key = getpass.getpass("Enter API key for Google Gemini: ")
        if huggingface_api_key:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = huggingface_api_key

        # Initialize LLM
        llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=google_api_key)
        
        # Initialize Embedding Model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   - Using device: {device}")
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large-instruct",
            model_kwargs={'device': device, 'token': huggingface_api_key}
        )
        
        print("   ✅ Models initialized successfully.")
        return llm, embeddings

    @staticmethod
    def setup_vector_store(embeddings):
        """Sets up a local Chroma vector store."""
        print("\n🗄️ Setting up Chroma vector store...")
        
        try:
            # Create or load a Chroma DB (persistent, so it survives restarts)
            # persist_directory = "./chroma_db"
            vector_store = Chroma(
                collection_name="rag_embeddings",
                embedding_function=embeddings,
                persist_directory="./chroma_db"  # Chroma will auto-save here
            )
            print("   ✅ Chroma vector store ready.")
            return vector_store, "rag_embeddings"
        except Exception as e:
            print(f"   ❌ Failed to initialize Chroma: {e}", file=sys.stderr)
            return None, None


    @staticmethod
    def load_and_embed_documents(vector_store, pdf_directory, embeddings):
        """Loads and embeds all PDFs (text-based + scanned) into Chroma."""
        print(f"\n📄 Syncing documents from directory: {pdf_directory}")

        try:
            pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]
        except FileNotFoundError:
            print(f"   - Directory not found: {pdf_directory}")
            return vector_store

        all_docs = []

        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            print(f"   - Loading: {pdf_file}")

            text_content = ""
            docs = []

            try:
                # 1) Try normal text extraction
                loader = PyMuPDFLoader(pdf_path)
                docs = loader.load()
                text_content = " ".join([d.page_content for d in docs]).strip()

                if not text_content:  # If empty, fallback to OCR
                    raise ValueError("No text found, using OCR")
            except Exception as e:
                print(f"     ⚠️ Using OCR for {pdf_file} (reason: {e})")

                try:
                    # 2) Convert pages to images & OCR
                    pages = convert_from_path(pdf_path)
                    ocr_text = ""
                    for page in pages:
                        ocr_text += pytesseract.image_to_string(page) + "\n"

                    docs = [Document(
                        page_content=ocr_text,
                        metadata={"source_file": pdf_file, "source": pdf_path}
                    )]
                except Exception as ocr_err:
                    print(f"     ❌ OCR failed for {pdf_file}: {ocr_err}")
                    continue

            all_docs.extend(docs)

        if not all_docs:
            print("   - No new documents to add.")
            return vector_store

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        all_splits = text_splitter.split_documents(all_docs)

        # Add instruction prefix
        docs_to_embed = [
            Document(
                page_content=f"Represent this document for semantic search: {doc.page_content}",
                metadata=doc.metadata,
            ) for doc in all_splits
        ]

        # Add to Chroma
        vector_store.add_documents(docs_to_embed)

        print(f"   ✅ Added {len(all_splits)} chunks from {len(pdf_files)} PDF(s).")
        return vector_store


    @staticmethod
    def create_rag_chain(vector_store, llm):
        """Creates the full RAG chain."""
        print("\n🔗 Creating RAG chain...")
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. You answer questions based only on the provided context. If the answer is not in the context, say 'I don't know'. And answer only in language the user asked the question."),
            ("user", "Question: {question}\n\nContext: {context}")
        ])

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("   ✅ RAG chain created.")
        return rag_chain

    @staticmethod
    def main():
        """Main function to run the RAG application."""
        vector_store = None  # Ensure vector_store is always defined
        try:
            llm, embeddings = RAGBOT.initialize_models()
            vector_store, collection_name = RAGBOT.setup_vector_store(embeddings)


            if not vector_store:
                print("\nExiting due to database connection failure.", file=sys.stderr)
                return


            # Point to the directory containing your PDFs
            pdf_directory = "pdfs"
            vector_store = RAGBOT.load_and_embed_documents(vector_store, "pdfs", embeddings)

            rag_chain = RAGBOT.create_rag_chain(vector_store, llm)

            print("\n" + "="*60)
            print("🤖 RAG system is ready. Ask a question or type 'exit' to quit.")
            print("="*60)

            while True:
                question = input("\nAsk a question: ")
                if question.lower() in ['exit', 'quit']:
                    print("👋 Goodbye!")
                    break
                
                print("\nThinking...\n")
                response = ""
                for chunk in rag_chain.stream(question):
                    print(chunk, end="", flush=True)
                    response += chunk
                print("\n")

        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    RAGBOT.main()