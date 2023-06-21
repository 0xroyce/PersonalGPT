# Import necessary libraries
import os
from dotenv import load_dotenv
import streamlit as st
import openai
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import Chroma

# Load .env file
load_dotenv()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Set persist flag to cache & reuse the model to disk (for repeated queries on the same data)
PERSIST = False

# Get user query input
query = st.text_input('Enter your query')

# Check if persist flag is set and persist directory exists
if PERSIST and os.path.exists("persist"):
    st.write("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    from langchain.indexes.vectorstore import VectorStoreIndexWrapper
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    # Load data from text file
    loader = TextLoader('data.txt')
    # This code can also import folders, including various filetypes like PDFs using the DirectoryLoader.
    # loader = DirectoryLoader(".", glob="*.txt")
    if PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])

# Create a RetrievalQA chain
chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo-0613"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

# Run query when button is clicked
if st.button('Run Query'):
    try:
        result = chain.run(query)
        st.write(result)
    except Exception as e:
        st.write("An error occurred while processing your request. Please try again later.")
        print(e)  # Log the error for debugging