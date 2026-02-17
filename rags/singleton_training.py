from utils import get_env_var
from langchain_community.vectorstores import Chroma
from langchain_classic.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from rags.vetorial_db import results_by_chromadb
from rags.etls import etl_pdf_process


class RagSingletonTraining:
    """
    Classe responsável por gerenciar o treinamento do agente RAG em um ambiente singleton.
    O objetivo é garantir que haja apenas uma instância do agente e do armazenamento de sessão durante o treinamento, evitando conflitos e garantindo consistência.
    """

    __instance: "RagSingletonTraining" = None
    __VECTOR_STORE: Chroma = None
    __QA_LLM: ChatGoogleGenerativeAI = None
    __DOCUMENTS: list[Document] = None

    def __new__(cls):
        """
        Implementação do padrão singleton para garantir que apenas uma instância da classe seja criada.
        """

        if cls.__instance is None:
            cls.__instance = super(RagSingletonTraining, cls).__new__(cls)

            print("Iniciando treinamento RAG")

            GEMINI_API_KEY = get_env_var("GEMINI_API_KEY")
            cls.__QA_LLM = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                temperature=0.1,
                api_key=GEMINI_API_KEY
            )

            GEMINI_API_KEY = get_env_var("GEMINI_API_KEY")
            embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=GEMINI_API_KEY
            )

            summary_enabled = str(get_env_var("RAG_SUMMARY_ENABLED", "false")).lower() in {"1", "true", "yes"}
            llm_for_summary = cls.__QA_LLM if summary_enabled else None

            documents = etl_pdf_process(llm_for_summary)

            required_metadata_defaults = {
                "id_doc": "N/A",
                "source": "N/A",
                "page_number": "N/A",
                "categoria": "N/A",
                "id_produto": "N/A",
                "preco": "N/A",
                "timestamp": "N/A",
                "data_owner": "N/A",
            }

            for doc in documents:
                metadata = doc.metadata or {}
                for key, default_value in required_metadata_defaults.items():
                    if key not in metadata:
                        metadata[key] = default_value
                    else:
                        value = metadata[key]
                        if hasattr(value, "item"):
                            metadata[key] = value.item()

                doc.metadata = metadata

            cls.__DOCUMENTS = documents

            cls.__VECTOR_STORE = results_by_chromadb(cls.__DOCUMENTS, embeddings)

        return cls.__instance

    def get_vector_store(self) -> Chroma:
        return self.__VECTOR_STORE

    def get_qa_llm(self) -> ChatGoogleGenerativeAI:
        return self.__QA_LLM

    def get_documents(self) -> list[Document]:
        return self.__DOCUMENTS