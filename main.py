""" Implemented CV Analyser using OpenAI's GPT-4 model and Pinecone for similarity search. 
 The user can upload multiple CVs in PDF or DOCX format, and the app will extract the text from the CVs using Google Vision API 
 for scanned images. The extracted text is then summarised using the GPT-4 model. The summarised CVs are stored in Pinecone 
 for similarity search. The user can ask questions about the CVs, and the app will retrieve the most relevant CVs based on 
 the question and generate a response using the RAG model. 
 The response is displayed to the user. The app uses Streamlit for the frontend and Python for the backend.
   The app is deployed on Streamlit Sharing."""

import os
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import streamlit as st
import pdfplumber
import tempfile
import io                                                           # Importing the required libraries
import requests
import base64
# from docx import Document
from dotenv import load_dotenv
from PIL import Image
from langchain_openai import OpenAI
import uuid
import json
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Keys 
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CLOUD_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")
VISION_URL = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_CLOUD_API_KEY}"

# Defined Functions 

def extract_text_from_pdf(file):
    extracted_text = ""                                         # Extracting text from PDF using pdfplumber and Vision API for scanned images.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.read())
        temp_filepath = temp_file.name

    with pdfplumber.open(temp_filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            
            if text and text.strip():
                extracted_text += text + "\n"
            else:
                image = page.to_image().original
                extracted_text += extract_text_from_image(image) + "\n"
    os.remove(temp_file.name)
    return extracted_text

def extract_text_from_docx(file):                       # Extracting text from a DOCX file
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_image(image):                     # Extracting text from an image using Google Vision API (OCR)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    content = img_byte_arr.getvalue()
    
    encoded_image = base64.b64encode(content).decode("utf-8")
    request_data = {
        "requests": [
            {
                "image": {"content": encoded_image},
                "features": [{"type": "TEXT_DETECTION"}],
            }
        ]
    }
    response = requests.post(VISION_URL, json=request_data)
    if response.status_code == 200:
        result = response.json()
        return result["responses"][0].get("fullTextAnnotation", {}).get("text", "No text found")
    else:
        return f"Error: {response.json()}"
    

def summarise_text(text):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an at analysing data from resumes and CVs. Given the following unstructured data, can you provide a summary of the CV with all the details included. Return it as a list with only two elements name and summary as a pargraph. Try to include personal details in the summary as well if mentioned."},
            {
                "role": "user",
                "content": text
            }
        ]
    )
    output_text = completion.choices[0].message.content
    output_text = output_text.replace("*", "").replace("\n", " ")   
    return output_text

def prepare_doc(final_results):
    if not final_results:
        return []
    final_list = []
    for result in final_results:
        temp ={}
        parsed_data = json.loads(result)
        name = parsed_data["name"]
        summary = parsed_data["summary"]
        temp["name"] = name
        temp["summary"] = summary
        final_list.append(temp)
    return final_list

def create_document(name, paragraph):                      # Creating a document object for each extracted CV
    return Document(
        page_content=paragraph,
        metadata={"id": str(uuid.uuid4()), "name": name}
    )


llm = init_chat_model("gpt-4o-mini", model_provider="openai")           # Initialising the OpenAI model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")           # Initialising the OpenAI embeddings

def generate_summaries():
    final_results = []

    st.title("CV - Analyser")
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX files", 
        type=["pdf", "docx"], 
        accept_multiple_files=True
    )

    if st.button("Submit"):
        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) uploaded successfully! Processing started...")
            
            for uploaded_file in uploaded_files:
                st.write(f"Processing: {uploaded_file.name}")

                if uploaded_file.name.endswith(".pdf"):
                    extracted_text = extract_text_from_pdf(uploaded_file)
                    summarised_text = summarise_text(extracted_text)
                    print("Summarised Text", summarised_text)
                    final_results.append(summarised_text)
                elif uploaded_file.name.endswith(".docx"):
                    extracted_text = extract_text_from_docx(uploaded_file)
                    summarised_text = summarise_text(extracted_text)
                    final_results.append(summarised_text)
                else:
                    extracted_text = "Unsupported file format."
                st.text_area(f"Summarised Text - {uploaded_file.name}", summarised_text, height=200)  # Displaying the summarised CV in the Streamlit app
        else:
            st.warning("No files uploaded. Please upload at least one file.")
    prepared_doc = prepare_doc(final_results)
    return prepared_doc

final_results = generate_summaries()
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))                # Initialising the Pinecone client and creating an index for storing the CV embeddings
index_name = "cvanalyser-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=spec
    )

index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
documents = []
for result in final_results:
    doc = create_document(result['name'], result['summary'])      # Each CV is converted to a document object and added to the Pinecone index for similarity search 
    documents.append(doc)
vector_store.add_documents(documents)

prompt = hub.pull("rlm/rag-prompt")                          # Initialising the RAG model for retrieving relevant documents

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])   # Building a state graph for the retrieval and generation process
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

user_input = st.text_input("Ask a question")                    # Taking user input for asking questions
if st.button("Ask"):
    response = graph.invoke({"question": user_input})
    st.write(response["answer"])
