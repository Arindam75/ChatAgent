import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.document_loaders import DirectoryLoader, TextLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import glob


class CustomEmbeddings:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", encode_kwargs=None):

        if encode_kwargs is None:
            encode_kwargs = {}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs=encode_kwargs
        )
        self.db = None

    def get_embedding(self, text):
        return self.embeddings.embed_query(text)

    def embed_data(self,
                   data_path,
                   db_path,
                   chunk_size=1000,
                   chunk_overlap=100):

        if not glob.glob(db_path + "\\chroma.sqlite3"):
            print("Vector Database Not Found !! Creating New Database")
            if not os.path.isdir(data_path):
                raise Exception("Data Path Not Found !!")
            text_splitter = RecursiveCharacterTextSplitter(
                                                           chunk_size=chunk_size,
                                                           chunk_overlap=chunk_overlap
                                                           )

            loader = DirectoryLoader(data_path,
                                     glob="**/*.txt",  # Recursively load all .txt files
                                     loader_cls=TextLoader)
            docs = loader.load_and_split(text_splitter=text_splitter)

            self.db = Chroma.from_documents(documents=docs,
                                            embedding=self.embeddings,
                                            persist_directory=db_path)
            self.db.persist()

        else:
            print("Vector Database Found !! Loading")
            self.db = Chroma(persist_directory=db_path,
                             embedding_function=self.embeddings)


class RagFusion:
    def __init__(self, model=None):
        self.llm = self.load_model()
        self.query_pipeline = self.fusion_query_pipeline()

    def load_model(self):
        
        return ChatOpenAI(model_name="gpt-4o-mini")

    def fusion_query_pipeline(self):

        fusion_prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful assistant that generates multiple queries based on a single input query."),
                        ("user",
                        ''' 
                         Generate five search queries related to: {question}\n. Follow the following guidelines:
                        - The first query should be the original input query.
                        - The search queries should be related to the original input query.
                        - Do not provide any further description of the query or question.
                        '''
                        )
                        ])
        query_pipeline = (
            fusion_prompt 
            | self.llm 
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
            )

        return query_pipeline
