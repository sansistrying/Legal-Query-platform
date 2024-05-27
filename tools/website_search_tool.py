from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
import requests
from crewai_tools import BaseTool
import csv
import datetime
import pandas as pd
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI

class website_search(BaseTool):
    # Define a class for the website search tool, inheriting from BaseTool
    name: str = "website reader"
    description: str = "Access the documents and answer the query"

    def _run(self, query: str) -> str:
        # Step 1: Fetch the sitemap XML and parse it
        url = "https://www.mlaw.gov.sg/sitemap.xml"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'lxml')

        # Step 2: Extract URLs from the sitemap XML
        url_list = soup.find_all('url')
        urls = []

        # Step 3: Extract URLs and filter based on last modified date
        for url in url_list:
            if '.pdf' in url.loc.get_text(): break

            if 'lastmod' in [tag.name for tag in url]:
                date = url.lastmod.get_text()
                date = datetime.datetime(int(date[:4]), int(date[5:7]), int(date[8:10]), int(date[11:13]))

                today = datetime.datetime.today()
                yesterday = today - datetime.timedelta(days=1)

                if date > yesterday:
                    urls.append(str(url.loc.get_text()))

        # Step 4: Load web documents from URLs
        print(len(urls))
        docs = [WebBaseLoader(url).load() for url in urls]

        # Step 5: Load or create a DataFrame for storing website data
        df = pd.read_csv("website_data.csv")
        df = df.drop('Unnamed: 0', axis=1)

        # Step 6: Update or append data to the DataFrame
        for url in urls:
            if url in df['URL']:
                for i in range(len(df['URL'])):
                    if df.loc[i, 'URL'] == url:
                        df.loc[i, 'Data'] = docs[i]
                        break
            else:
                df.loc[len(df)] = {'URL': url, 'Data': docs[urls.index(url)]}

        # Step 7: Upload the updated DataFrame to a CSV file
        df.to_csv("website_data.csv")

        # Step 8: Load documents from the updated DataFrame
        docs_list = list(df['Data'])

        # Step 9: Convert documents to SimpleDocument instances
        class SimpleDocument:
            def __init__(self, page_content, metadata=None):
                self.page_content = page_content
                self.metadata = metadata or {}

        documents = [SimpleDocument(page_content=" ".join(row)) for row in docs_list]

        # Step 10: Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)

        # Step 11: Convert text chunks to embeddings
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_documents(texts, embeddings)

        # Step 12: Load question answering chain and find relevant documents
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        docs = docsearch.similarity_search(query)

        # Step 13: Run question answering chain on input documents and query
        return chain.run(input_documents=docs, question=query)
