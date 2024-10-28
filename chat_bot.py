import streamlit as st
import os
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["CHROMA_DB_DIR"] = "./chroma_db"


st.title("Rapids AI Chat-Bot")

# Load and prepare the PDF document
loader = PyPDFLoader("Rapids_AI_Text_Data.pdf")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Create the vector store
vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Set up the chatbot model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

# Set up system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if st.button("Clear Chat History"):
    st.session_state.chat_history = []

# Chat input from user
query = st.chat_input("Ask Me Anything: ") 

if query:
    
    st.session_state.chat_history.append(("User", query))
    
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    
    
    answer = response["answer"]
    st.session_state.chat_history.append(("Bot", answer))
    
   
    chromadb.api.client.SharedSystemClient.clear_system_cache()

# Display chat history with icons
for sender, message in st.session_state.chat_history:
    if sender == "User":
        st.markdown(f"👤 **You:** {message}")
    else:
        st.markdown(f"🤖 **Bot:** {message}")
