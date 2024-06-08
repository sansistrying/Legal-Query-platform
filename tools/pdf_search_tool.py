from crewai_tools import BaseTool
import os
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Qdrant
from langchain.prompts import PromptTemplate

class PDF_search(BaseTool):
    name: str = "PDF reader"  # Name of the tool
    description: str = "Access the documents and answer the query, also give the corresponding document name."  # Description of the tool

    def _run(self, user_query: str) -> str:
        # Load all PDF documents from the specified directory
        loader = PyPDFDirectoryLoader(r"C:\Users\sansi\OneDrive\Desktop\Collyear_Law_Project\pdf")
        documents = loader.load()

        # Split the loaded documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(documents)

        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings()

        # Create a Qdrant vector store from the document splits
        db = Qdrant.from_documents(all_splits, embeddings, location=":memory:", collection_name="all_splits", distance_func="Dot")

        # Define the prompt template for the QA task
        prompt_template = """Text: {context}

        Question: {question}

        Answer the question based on the PDF Document provided. If the text doesn't contain the answer, reply that the answer is not available.
        Do Not Hallucinate"""

        # Create a PromptTemplate object with the defined template
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "query"]
        )
        chain_type_kwargs = {"prompt": PROMPT}  # Additional keyword arguments for the chain

        # Initialize the RetrievalQA chain with the specified LLM, retriever, and prompt
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125", api_key="your-api-key"),
            chain_type="stuff",
            input_key='query',
            retriever=db.as_retriever(),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )

        # Pass the user query to the QA object and get the result
        result = qa({'query': user_query})
        return result  # Return the result
