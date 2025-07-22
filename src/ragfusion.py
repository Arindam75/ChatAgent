import os
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import textwrap
from langchain.load import dumps, loads
from langchain.schema.runnable import RunnableLambda
from huggingface_hub import snapshot_download

def wrap_text(text: str, 
              width: int=90) -> str:
    """
    Wraps each line of the input text to a specified width.

    This function takes a multi-line string and wraps each line to the given width,
    preserving the original line breaks.

    Args:
        text (str): The input text to be wrapped.
        width (int, optional): The maximum width of each wrapped line. Defaults to 90.

    Returns:
        str: The text with each line wrapped to the specified width.
    """

    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def reciprocal_rank_fusion(results: list[list], 
                           k: int=60) -> list: 
    """
    Fuse multiple lists of documents by their reciprocal rank.

    This method takes in multiple lists of documents and fuses them into a single list
    by computing the reciprocal rank of each document in each list. The reciprocal rank
    is the inverse of the rank of the document in the list, scaled by k so that the
    highest ranked document has a score of 1 and the lowest ranked document has a score
    of 1/k.

    Args:
        results (list[list]): A list of lists of documents to be fused.
        k (int, optional): The scaling factor for the reciprocal rank. Defaults to 60.

    Returns:
        list: A list of (document, score) tuples, where the score is the reciprocal rank
        of the document in the fused list.
    """
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        loads(doc)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

class RagFusion:
    def __init__(self,
                 llm: str = None,
                 db_path: str = None,
                 emb_model_name: str="BAAI/bge-base-en-v1.5"):
        
        """
        Args:
            llm: The LLM to use for generating queries.
            db_path: The path to the Chroma database.
            emb_model_name: The name of the model to use for generating embeddings.

        Raises:
            ValueError: If llm, db_path, or emb_model_name is None.
        """

        if llm is None:
            raise ValueError("llm is required")
        
        if db_path is None:
            raise FileNotFoundError("db_path is required")
        
        if emb_model_name is None:
            raise ValueError("emb_model_name is required")

        self.llm = self.__load_llm(llm)
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(model_name=emb_model_name)
        self.vectorstore = self.__build_vectorstore()

    def __load_llm(self, 
                   llm: str = None):

        return ChatGoogleGenerativeAI(model=llm)
    
    def __build_vectorstore(self):

        """
        Builds a Chroma vectorstore instance.

        This method creates an instance of Chroma vectorstore using the
        embeddings model and the database path provided in the class
        constructor.

        Returns:
            langchain.vectorstores.Chroma: A Chroma vectorstore instance.
        """

        if not (Path(self.db_path) / "chroma.sqlite3").exists():
            print("downloading vector dataset...")
            db_path = os.path.abspath(self.db_path)
            snapshot_download(
                                repo_id="Arindam1975/Singapore",
                                repo_type="dataset",
                                token=os.environ["HF_API_KEY"],
                                local_dir=db_path,
                            )
        vectorstore = Chroma(
                            persist_directory=self.db_path,
                            embedding_function=self.embeddings,
                            collection_name="singapore_tourism"
                        )

        return vectorstore

    def get_docs_with_scores(self,query):
        """Custom retriever that returns docs with scores"""
        results = self.vectorstore.similarity_search_with_score(query, k=3)
        return [(doc, 1 - score) for doc, score in results if (1 - score) > 0.75]    

    def fusion_chain(self, question):

        """
        Fusion chain that generates multiple queries based on a single input query.

        This method creates a Langchain pipeline that generates multiple search queries
        based on a single input query. The pipeline uses the provided LLM to generate
        related queries, and then uses the Chroma vectorstore to find the top 3 matching
        documents for each query. The final output is a list of (document, score) tuples.

        Args:
            question (str): The input query to generate related queries from.

        Returns:
            list: A list of (document, score) tuples, where the score is the reciprocal rank
            of the document in the fused list.
        """
        fusion_prompt = ChatPromptTemplate.from_messages([ 
            ("system", "You are a helpful assistant that generates multiple queries based on a single input query."),
            ("user", ''' Generate five search queries related to: {question}\n. Follow the following guidelines:
                        - The first query should be the original input query.
                        - The search queries should be related to the original input query.
                        - Do not provide any further description of the query or question.
                     '''
            )
        ])

        fusion_search_chain = (
            fusion_prompt 
            | self.llm 
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
            | RunnableLambda(self.get_docs_with_scores).map()
            | reciprocal_rank_fusion
        )
        return fusion_search_chain.invoke({"question": question})