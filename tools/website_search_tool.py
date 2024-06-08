from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
import requests
from crewai_tools import BaseTool
import csv
import datetime
import pandas as pd
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests
from bs4 import BeautifulSoup
import datetime

class website_search(BaseTool):
    name: str = "website reader"  # Name of the tool
    description: str = "Access the documents and answer the query"  # Description of the tool

    def _run(self, query: str) -> str:
        url = "https://www.mlaw.gov.sg/sitemap.xml"  # URL to fetch the sitemap
        response = requests.get(url)  # Making a request to the sitemap URL
        soup = BeautifulSoup(response.content, features='xml')  # Parsing the XML response
        url_list = soup.find_all('url')  # Finding all URL tags in the sitemap

        today = datetime.datetime.today()  # Getting today's date
        yesterday = today - datetime.timedelta(days=1)  # Calculating yesterday's date

        # Filtering URLs that are not PDFs and have been modified in the last day
        urls = [url.loc.get_text() for url in url_list if ('.pdf' not in url.loc.get_text() and
                                                          'lastmod' in [tag.name for tag in url] and
                                                          datetime.datetime.strptime(url.lastmod.get_text()[:19], "%Y-%m-%dT%H:%M:%S") > yesterday)]

        # Loading existing website data from a CSV file
        df = pd.read_csv(r"C:\Users\sansi\Downloads\website_data.csv").drop('Unnamed: 0', axis=1)
        url_data_map = {row['URL']: row['Data'] for _, row in df.iterrows()}  # Creating a map of URLs to data

        if urls:  # If there are URLs to process
            with ThreadPoolExecutor() as executor:  # Using ThreadPoolExecutor for concurrent loading
                docs = list(executor.map(lambda u: WebBaseLoader(u).load(), urls))  # Loading documents concurrently

            for i, url in enumerate(urls):  # Updating the DataFrame with new data
                if url in url_data_map:
                    df.loc[df['URL'] == url, 'Data'] = docs[i]  # Update existing entry
                else:
                    df.loc[len(df)] = {'URL': url, 'Data': docs[i]}  # Add new entry

            df.to_csv("website_data.csv", index=False)  # Save updated DataFrame to CSV

        docs_list = df['Data'].tolist()[:100]  # Limiting the documents list to the first 100 entries

        # Defining a simple document class
        class SimpleDocument:
            def __init__(self, page_content, metadata=None):
                self.page_content = page_content  # Page content of the document
                self.metadata = metadata or {}  # Metadata of the document (default to empty dict)

        # Creating SimpleDocument instances from the documents list
        documents = [SimpleDocument(page_content=" ".join(row)) for row in docs_list]

        # Splitting the documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()  # Initializing OpenAI embeddings
        docsearch = FAISS.from_documents(texts, embeddings)  # Creating a FAISS vector store from the documents
        chain = load_qa_chain(OpenAI(), chain_type="stuff")  # Loading a QA chain using OpenAI

        docs = docsearch.similarity_search(query)  # Performing a similarity search on the documents

        return chain.run(input_documents=docs, question=query)  # Running the QA chain and returning the answer
