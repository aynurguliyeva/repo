import os
import requests
from PyPDF2 import PdfReader
from crewai import Agent  # Use Agent for behavior, not Pydantic
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import streamlit as st
from pydantic import BaseModel  # Only for user input data validation



# Load environment variables
load_dotenv()



# Retrieve the API key from environment
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Please check your .env file.")

# Step 1: Define a simple Pydantic model for input validation
class IngestionAgentInput(BaseModel):
    goal: str
    backstory: str

# Step 2: Define the IngestionAgent that will use the goal and backstory
class IngestionAgent:
    def __init__(self, goal: str, backstory: str):
        # Initialize the agent with goal and backstory
        self.goal = goal
        self.backstory = backstory

    def process_pdf(self, pdf_file):
        # Extract text from the PDF file
        with open(pdf_file, 'rb') as file:
            pdf_reader = PdfReader(file)
            text = ""
            for page in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page].extract_text()
        
        # Create embeddings from the extracted text
        embeddings = OpenAIEmbeddings()
        vector = embeddings.embed_documents([text])
        
        # Store the vector in Chroma (vector database)
        db = Chroma.from_documents([text], embeddings)
        
        # Persist the database to disk
        db.persist("db_path")  # Save the database to "db_path" folder
        
        # Return a simple message that the ingestion is complete
        return f"Ingestion Complete: The document has been ingested with goal: {self.goal}"

class QuestionAnsweringAgent:
    def answer_question(self, query):
        # Load the Chroma DB from the persisted directory
        db = Chroma("db_path", OpenAIEmbeddings())  # Initialize the Chroma object from the saved directory
        relevant_docs = db.similarity_search(query)
        
        # Groq API URL and headers (authentication)
        groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {'Authorization': f"Bearer {os.getenv('GROQ_API_KEY')}", "Content-Type": "application/json"}
        
        # Prepare the payload for the Groq API
        payload = {
            "model": "llama3-8b-8192",  # Replace with your desired model
            "messages": [{"role": "user", "content": query}]
        }
        
        # Debugging: Print headers and payload
        print("Headers:", headers)
        print("Payload:", payload)
        
        # Send the query to the Groq API
        try:
            response = requests.post(groq_api_url, json=payload, headers=headers)
            print("Response Status Code:", response.status_code)
            print("Response Content:", response.text)  # Log full response for debugging
            
            if response.status_code == 200:
                answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No answer found.")
                return f"Answer: {answer}"
            else:
                return f"Error: {response.status_code} - {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Request failed: {e}"

# Step 4: Define the Streamlit user interface
def streamlit_interface():
    st.title("Personalized Learning Assistant")

    # Upload section to add extra materials (PDF files)
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    # Select the LLM (Groq by default)
    llm_choice = st.selectbox("Select LLM", ["Groq", "OpenAI", "HuggingFace"])
    
    if uploaded_file is not None:
        st.write("Processing your PDF document...")
        
        # Get goal and backstory from user input
        goal = st.text_input("Enter the goal for the ingestion process:", value="Extract information from PDF.")
        backstory = st.text_area("Enter the backstory for the agent:", value="This agent processes PDF files for a learning assistant.")
        
        if goal and backstory:
            # Validate the input with Pydantic
            try:
                # Use Pydantic to validate goal and backstory
                agent_input = IngestionAgentInput(goal=goal, backstory=backstory)
                
                # Initialize the IngestionAgent with the goal and backstory
                ingestion_agent = IngestionAgent(goal=agent_input.goal, backstory=agent_input.backstory)
                ingestion_result = ingestion_agent.process_pdf(uploaded_file)
                st.write(ingestion_result)  # Display the ingestion result
            except Exception as e:
                st.write(f"Validation Error: {e}")
        else:
            st.write("Please provide both the goal and backstory.")
    
    # Chatbot interface for answering user questions
    query = st.text_input("Ask a question")
    
    if query:
        st.write("Fetching the answer...")
        
        # Initialize the question answering agent
        qa_agent = QuestionAnsweringAgent()
        answer = qa_agent.answer_question(query)
        st.write(answer)  # Display the answer

# Step 5: Run the Streamlit app
if __name__ == "__main__":
    streamlit_interface()




