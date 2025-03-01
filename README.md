# CV-Analyser
# Overview

The CV Analyser is a web-based application that allows users to upload multiple CVs in PDF or DOCX format. The app extracts text from these documents using Google Vision API (OCR), processes the text using GPT-4 for summarization, and stores the summarized CVs in Pinecone for similarity search.

Using Retrieval-Augmented Generation (RAG), the application allows users to ask questions about the CVs, retrieving the most relevant ones using LangChain and generating meaningful responses based on the extracted information. The app is built using Streamlit for the frontend and Python for backend processing.

# Features

- OCR and Text Extraction: Extracts text from PDFs and DOCX files using pdfplumber and Google Vision API (for scanned documents and images).

- Summarization: Uses GPT-4 to summarize extracted CV text, ensuring key details are retained.

- Vector Storage & Similarity Search: Summarized CVs are embedded using OpenAI Embeddings and stored in Pinecone for efficient retrieval.

- RAG-based Question Answering: Users can ask questions, and the system retrieves relevant CVs and generates responses using LangChain and GPT-4.

- Streamlit UI: A simple and interactive web interface for uploading files, viewing summaries, and querying CVs.

- Deployment: Hosted on Streamlit Sharing for easy accessibility.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Pip

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Sami7622/cv-analyser.git
   cd cv-analyser
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the application:
```bash
streamlit run main.py
```


## Configuration
This project uses API keys for Google Vision and OpenAI. Store them as environment variables:
```bash
export GOOGLE_API_KEY="your-google-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

Alternatively, create a `.env` file:
```env
GOOGLE_API_KEY=your-google-api-key
OPENAI_API_KEY=your-openai-api-key
```


