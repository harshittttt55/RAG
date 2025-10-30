RAG Project Setup & Usage

Note: 
Main pdf file is in pdfs section but ensure that the pdfs should be consist computer text with less images and avoid the scanned images.

1️⃣ Clone the repository
git clone <your-repo-url>
cd rag_project

2️⃣ Create & activate virtual environment

Windows:

python -m venv rag_env
.\rag_env\Scripts\activate


Linux / Mac:

python3 -m venv rag_env
source rag_env/bin/activate

3️⃣ Install Python dependencies
pip install -r requirements.txt

4️⃣ Install system dependencies
Poppler (for pdf2image):

Windows: Download Poppler
 → add bin/ folder to PATH.

Linux:

sudo apt install poppler-utils

Tesseract OCR (for pytesseract):

Windows: Download Tesseract
 → add to PATH.

Linux:

sudo apt install tesseract-ocr

5️⃣ Add your API keys

Create a .env file in the project root:

GEMINI_API_KEY=<your-google-gemini-api-key>
HUGGINGFACE_API_TOKEN=<your-huggingface-token>

6️⃣ Prepare your PDFs

Place all PDFs to embed/search in:

rag_project/pdfs/

7️⃣ Run the RAG system in console (optional)
python Rag_Pipeline.py


Interactive console mode.

Type your questions and get answers.

8️⃣ Run the Streamlit Chat UI
streamlit run streamlit_app.py


Left side: ask questions & see answers

Right side: search history

First load may take a few seconds while models and Chroma DB initialize.

9️⃣ Optional: Clear search history

Close & reopen Streamlit app

Or implement a “Clear History” button (future enhancement)

Preview:

<img width="1917" height="865" alt="Image" src="https://github.com/user-attachments/assets/dae8d723-06ba-4158-a4ad-54764a906774" />




