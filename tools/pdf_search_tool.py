from crewai_tools import BaseTool
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS

class PDF_search(BaseTool):
    # Define a class for the PDF search tool, inheriting from BaseTool
    name: str = "PDF reader"
    description: str = "Access the documents and answer the query"

    def _run(self, query: str) -> str:
        # Step 1: Iterate through each folder in the specified directory
        for folder in os.listdir(r"C:\Users\sansi\Downloads\Collyear Law"):

            raw_text = ''  # Initialize variable to store raw text from PDFs

            # Step 2: Iterate through each file in the folder
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)  # Get the full file path
                reader = PdfReader(file_path)  # Initialize PdfReader object

                # Step 3: Iterate through each page in the PDF
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()  # Extract text from the page
                    if text:
                        raw_text += text  # Append extracted text to raw_text

        # Step 4: Split the raw text into smaller chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        # Step 5: Convert text chunks to embeddings
        embeddings = OpenAIEmbeddings()

        # Step 6: Create FAISS index from text embeddings
        docsearch = FAISS.from_texts(texts, embeddings)

        # Step 7: Load question answering chain
        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        # Step 8: Search for relevant documents based on query
        docs = docsearch.similarity_search(query)

        # Step 9: Run question answering chain on input documents and query
        return chain.run(input_documents=docs, question=query)
