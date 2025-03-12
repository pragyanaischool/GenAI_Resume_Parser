import streamlit as st
#from backend.pdf_ingestion import load_split_pdf
#from backend.vector_store import create_vector_store
#from backend.analysis import analyze_resume
import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load and split the PDF document and return the documents and text chunks
def load_split_pdf(file_path):
    # Load the PDF document and split it into chunks
    loader = PyPDFLoader(file_path)  # Initialize the PDF loader with the file path
    documents = loader.load()  # Load the PDF document 

    # Initialize the recursive character text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,  # Set the maximum chunk size
        chunk_overlap=20,  # Set the number of overlapping characters between chunks
        separators=["\n\n", "\n", " ", ""],  # Define resume-specific separators for splitting
    )   

    # Split the loaded documents into chunks
    chunks = text_splitter.split_documents(documents)
    return documents, chunks

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize the Hugging Face embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def create_vector_store(chunks):
    # Store embeddings into the vector store
    vector_store = FAISS.from_documents(
        documents=chunks,  # Input chunks to the vector store
        embedding=embeddings  # Use the initialized embeddings model
    )
    return vector_store
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Set up Groq API key
load_dotenv()  # Load environment variables from .env file
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")  # Set the Groq API key from environment variables

# Initialize the ChatGroq model with the specified model name
llm = ChatGroq(model_name="mixtral-8x7b-32768")

def analyze_resume(full_resume, job_description):
    # Template for analyzing the resume against the job description
    template = """
    You are an AI assistant specialized in resume analysis and recruitment. Analyze the given resume and compare it with the job description. 
    
    Example Response Structure:
    
    **OVERVIEW**:
    - **Match Percentage**: [Calculate overall match percentage between the resume and job description]
    - **Matched Skills**: [List the skills in job description that match the resume]
    - **Unmatched Skills**: [List the skills in the job description that are missing in the resume]

    **DETAILED ANALYSIS**:
    Provide a detailed analysis about:
    1. Overall match percentage between the resume and job description
    2. List of skills from the job description that match the resume
    3. List of skills from the job description that are missing in the resume
    
    **Additional Comments**:
    Additional comments about the resume and suggestions for the recruiter or HR manager.

    Resume: {resume}
    Job Description: {job_description}

    Analysis:
    """
    prompt = PromptTemplate(  # Create a prompt template with input variables
        input_variables=["resume", "job_description"],
        template=template
    )

    # Create a chain combining the prompt and the language model
    chain = prompt | llm

    # Invoke the chain with input data
    response = chain.invoke({"resume": full_resume, "job_description": job_description})

    # Return the content of the response
    return response.content
# Main application including "Upload Resume" and "Resume Analysis" sections
def render_main_app():
    
    # Apply custom CSS to adjust the sidebar width
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            min-width: 25%;
            max-width: 25%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Moving the upload section to the sidebar
    with st.sidebar:
        st.header("Upload Resume")  # Header for the upload section
        
        # File uploader for PDF resumes
        resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

        # Text area for job description input
        job_description = st.text_area("Enter Job Description", height=300)

        if resume_file and job_description:  # Check if both inputs are provided
            # Create a temporary directory if it doesn't exist
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)

            # Save the uploaded file to the temporary directory
            with open(os.path.join(temp_dir, resume_file.name), "wb") as f:
                f.write(resume_file.getbuffer())
            
            # Load and split the PDF file into documents and chunks
            resume_file_path = os.path.join("temp", resume_file.name)
            resume_docs, resume_chunks = load_split_pdf(resume_file_path)

            # Create a vector store from the resume chunks
            vector_store = create_vector_store(resume_chunks)
            st.session_state.vector_store = vector_store  # Store vector store in session state
                
            # Remove the temporary directory and its contents
            shutil.rmtree(temp_dir)

            # Button to begin resume analysis
            if st.button("Analyze Resume", help="Click to analyze the resume"):
                # Combine all document contents into one text string for analysis
                full_resume = " ".join([doc.page_content for doc in resume_docs])
                # Analyze the resume
                analysis = analyze_resume(full_resume, job_description)
                # Store analysis in session state
                st.session_state.analysis = analysis    
        else:
            st.info("Please upload a resume and enter a job description to begin.")

    # Display the analysis result if it exists in session state 
    if "analysis" in st.session_state:
        st.header("Resume-Job Compatibility Analysis")
        st.write(st.session_state.analysis)
    else:
        st.header("Welcome to the Ultimate Resume Analysis Tool!")
        st.subheader("Your one-stop solution for resume screening and analysis.")
        st.info("Do you want to find out the compatibility between a resume and a job description? So what are you waiting for?")

        todo = ["Upload a Resume", "Enter a Job Description", "Click on Analyze Resume"]
        st.markdown("\n".join([f"##### {i+1}. {item}" for i, item in enumerate(todo)]))
