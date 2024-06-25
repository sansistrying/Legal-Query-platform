from crewai_tools import BaseTool
import os
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, RecursiveJsonSplitter
from langchain_community.llms import OpenAI
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Qdrant

from langchain.prompts import PromptTemplate

# from llama_parse import LlamaParse
# import joblib
# from llama_index.core import SimpleDirectoryReader
from langchain_community.document_loaders import UnstructuredMarkdownLoader, JSONLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from langchain_ai21 import AI21Embeddings


from ai21 import AI21Client
from ai21.models import EmbedType

class PDF_search(BaseTool):
    name: str = "PDF reader"
    description: str = "Access the documents and answer the query, also give the corresponding document name only not the path."

    def _run(self, user_query: str) -> str:

        # loader = DirectoryLoader('./json', glob = "**/*.json", show_progress = True, loader_cls = JSONLoader, loader_kwargs = {'jq_schema':'.content', 'text_content': False})
        # json_data = loader.load()


        # splitter = RecursiveJsonSplitter(max_chunk_size=300)

        # # Recursively split json data - If you need to access/manipulate the smaller json chunks
        # json_chunks = splitter.split_json(json_data=json_data)


   #     model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')

       # embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        #embeddings = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")
        embeddings = AI21Embeddings()

        if "qdrant_db_pdfs" not in os.listdir():
            loader = PyPDFDirectoryLoader("/home/palak/Desktop/Collyear_Law_Project/pdf")
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
            all_splits = text_splitter.split_documents(documents)

            os.mkdir("./qdrant_db_pdfs")
            
            db = Qdrant.from_documents(all_splits, embeddings, path = "./qdrant_db_pdfs", collection_name = "pdfs", distance_func="Dot")

        else:
            client = QdrantClient(path = "./qdrant_db_pdfs")
            db = Qdrant(client = client, collection_name = "pdfs", embeddings=embeddings, )


        print('3')

        prompt_template = """Text: {context}

        Question: {question}

        Answer the question based on the PDF Document provided. If the text doesn't contain the answer, reply that the answer is not available.
        Do Not Hallucinate. Just give one accurate answer."""


        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "query"]
        )
        chain_type_kwargs = {"prompt": PROMPT}
        
                               
        qa = RetrievalQA.from_chain_type(llm = ChatGroq(temperature=0, model_name="llama3-70b-8192"),
                                        chain_type="stuff",
                                        input_key = 'query',
                                        retriever=db.as_retriever(),
                                        chain_type_kwargs=chain_type_kwargs,
                                        return_source_documents=True)


        # Pass the query to the qa object using the correct key
        result = qa({'query': user_query})
        return result
