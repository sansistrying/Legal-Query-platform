from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
import requests
from crewai_tools import BaseTool
import os
import datetime
import pandas as pd
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, RecursiveJsonSplitter
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests
from bs4 import BeautifulSoup
import datetime
from qdrant_client import QdrantClient

from langchain_ai21 import AI21Embeddings
from langchain_groq import ChatGroq
import faiss

from langchain_community.document_loaders import UnstructuredMarkdownLoader, JSONLoader, DirectoryLoader
from langchain.prompts import PromptTemplate

class website_search(BaseTool):
    name: str = "website reader"
    description: str = "Access the documents and answer the query"

    def _run(self, user_query: str) -> str:
        url = "https://www.mlaw.gov.sg/sitemap.xml"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, features='xml')
        url_list = soup.find_all('url')

        today = datetime.datetime.today()
        yesterday = today - datetime.timedelta(days=1)

        urls = [url.loc.get_text() for url in url_list if ('.pdf' not in url.loc.get_text() and
                                                          'lastmod' in [tag.name for tag in url] and
                                                          datetime.datetime.strptime(url.lastmod.get_text()[:19], "%Y-%m-%dT%H:%M:%S") > yesterday)]

        df = pd.read_csv(r"C:\Users\sansi\Downloads\Collyear_Law_proj\website_data.csv")
        #.drop('Unnamed: 0', axis=1)
        url_data_map = {row['URL']: row['Data'] for _, row in df.iterrows()}

        print(len(urls))

        if urls:
            with ThreadPoolExecutor() as executor:
                docs = list(executor.map(lambda u: WebBaseLoader(u).load(), urls))

            for i, url in enumerate(urls):
                if url in url_data_map:
                    df.loc[df['URL'] == url, 'Data'] = docs[i]
                else:
                    df.loc[len(df)] = {'URL': url, 'Data': docs[i]}

            df.to_csv("website_data.csv", index=False)

        docs_list = df['Data'].tolist()

        print(1)

        class SimpleDocument:
            def __init__(self, page_content, metadata=None):
                self.page_content = page_content
                self.metadata = metadata or {}

        documents = [SimpleDocument(page_content=" ".join(row)) for row in docs_list]


        # loader = DirectoryLoader('./website data', glob = "**/*.json", show_progress = True, loader_cls = JSONLoader, loader_kwargs = {'jq_schema':'.content', 'text_content': False})
        # json_data = loader.load()

        # print(1)
        # splitter = RecursiveJsonSplitter(max_chunk_size=300)

        # # Recursively split json data - If you need to access/manipulate the smaller json chunks
        # json_chunks = splitter.split_json(json_data=json_data)

        embeddings = AI21Embeddings()
        if "qdrant_db_pdfs" not in os.listdir():
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)

            os.mkdir("./qdrant_db_pdfs")
            
            db = Qdrant.from_documents(texts, embeddings, path = "./qdrant_db_website", collection_name = "website", distance_func="Dot")

        else:
            client = QdrantClient(path = "./qdrant_db_website")
            db = Qdrant(client = client, collection_name = "website", embeddings=embeddings, )

        print(2)

        prompt_template = """Text: {context}

        Question: {question}

        Answer the query based on the data in the db. If the answer cannot be found, say "Please provide more context" or "I don't know the answer based on the available data".
        
        Respond only once and do not generate additional outputs.
       
        Do Not Hallucinate."""

        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "query"]
        )
        chain_type_kwargs = {"prompt": PROMPT}
        
                               
        qa = RetrievalQA.from_chain_type(llm = ChatGroq(temperature=0, model_name="llama3-70b-8192"),
                                        chain_type="stuff",
                                        input_key = 'query',
                                        retriever=db.as_retriever(),
                                        chain_type_kwargs=chain_type_kwargs,
                                        )


        # Pass the query to the qa object using the correct key
        result = qa({'query': user_query})
        return result
