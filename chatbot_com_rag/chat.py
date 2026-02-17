from utils import load_environment_variables, get_env_var
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from guardrails_security import GuardrailsSecurity
from vetorial_db import results_by_chromadb
from utils import get_prompt
from etls import etl_pdf_process


class ChatbotSingleton:
    """
    Singleton para instância do chatbot.
    """

    _instance = None
    _session_store: dict[str, InMemoryChatMessageHistory] = {}

    def __new__(cls, *args, **kwargs):
        """
        Garante que apenas uma instância do chatbot seja criada (singleton).
        """

        if cls._instance is None:
            cls._instance = super(ChatbotSingleton, cls).__new__(cls)
            cls._instance._initialize(*args, **kwargs)

        return cls._instance

    def __init__(self, session_id: str) -> None:
        """
        Inicializa o histórico da sessão.
        """

        print(f"Criando nova instância do chatbot para sessão '{session_id}'...")

        if session_id not in self._session_store:
            self._session_store[session_id] = InMemoryChatMessageHistory()

        self.session = self._session_store[session_id]

        # Wrapper que injeta histórico e registra novas mensagens automaticamente.
        self.chain = RunnableWithMessageHistory(
            self.__rag_chain,
            lambda *args, **kwargs: self.session,  # Usa o histórico específico da sessão.
            input_messages_key="input",  # Chave do texto da pergunta.
            history_messages_key="chat_history",  # Onde o histórico é lido/escrito.
            output_messages_key="answer",  # Campo de resposta no output.
        )

    def _initialize(self, session_id: str) -> None:
        """
        Resumo simples do que o código faz:

        1. Carrega a chave da API do Gemini do ambiente.
        2. Cria um modelo de embeddings (para transformar texto em vetores).
        3. Cria um modelo de chat (para responder perguntas).
        4. Lê dados de PDF e de um “banco” (ETL), junta tudo e adiciona metadados padrão.
        5. Indexa os documentos em um banco vetorial (Chroma) para busca semântica.
        6. Quando recebe uma pergunta, ela é reescrita considerando o histórico.
        7. Busca os documentos mais relevantes.
        8. Monta um prompt com esses documentos e gera a resposta.
        9. Mostra a resposta e quantos documentos foram usados.
        """

        print(f"Inicializando chatbot com sessão '{session_id}'...")

        load_environment_variables()

        # Recupera a chave de API do Gemini do ambiente. Falha cedo se ausente.
        GEMINI_API_KEY = get_env_var('GEMINI_API_KEY')

        # Instancia o modelo de embeddings do Google para vetorizar texto.
        # Esses vetores são necessários para busca semântica no banco vetorial.
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",  # Modelo específico de embeddings.
            google_api_key=GEMINI_API_KEY  # Credencial exigida pela API.
        )

        # Instancia o LLM para respostas e para reescrever a pergunta com contexto.
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",  # Modelo leve/rápido para conversação.
            temperature=0.1,  # Baixa aleatoriedade para respostas mais consistentes.
            api_key=GEMINI_API_KEY  # Credencial exigida pela API.
        )

        # Executa ETL dos PDFs (pode usar o LLM para limpeza/extração).
        documents = etl_pdf_process(llm)

        # Metadados obrigatórios para o prompt dos documentos.
        # Valores padrão evitam KeyError quando a fonte não fornece algum campo.
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

        # Normaliza metadados (inclui defaults e converte tipos não serializáveis).
        for doc in documents:
            metadata = doc.metadata or {}
            for key, default_value in required_metadata_defaults.items():
                if key not in metadata:
                    metadata[key] = default_value
                else:
                    value = metadata[key]
                    # Converte objetos tipo NumPy scalar em tipos Python nativos.
                    if hasattr(value, "item"):
                        metadata[key] = value.item()

            doc.metadata = metadata

        # Indexa documentos no ChromaDB e retorna o repositório vetorial.
        vector_store = results_by_chromadb(documents, embeddings)

        # Prompt para reescrever a pergunta com base no histórico (sem responder).
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", get_prompt("contextualize_query.prompt.md")),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # Prompt principal de QA: usa contexto recuperado e histórico de conversa.
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", get_prompt("qa_system.prompt.md")),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # Prompt que define como cada documento aparece no contexto da resposta.
        document_prompt = PromptTemplate.from_template(get_prompt("doc_context.prompt.md"))

        # Recuperador semântico com top-k documentos mais relevantes.
        semantic_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        # Lexical retriever (BM25) para complementar a busca semântica, especialmente útil para termos específicos.
        lexical_retriever = BM25Retriever.from_documents(documents)
        lexical_retriever.k = 5  # Configura para retornar os 5 documentos mais relevantes.

        # Fazer o merge dos resultados dos dois recuperadores (semântico + lexical) para melhorar a cobertura.
        # O EnsembleRetriever combina os resultados de ambos, dando mais peso ao semântico.
        ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, lexical_retriever],
            weights=[0.7, 0.3]  # Dá mais peso ao recuperador semântico, mas ainda considera o lexical.
        )

        # Recuperador que reescreve a pergunta considerando o histórico.
        history_aware_retriever = create_history_aware_retriever(llm, ensemble_retriever, contextualize_q_prompt)

        # Cadeia de QA que insere documentos no prompt de resposta.
        # Junta os documentos no prompt e faz a resposta
        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt,
            document_prompt=document_prompt,
            document_variable_name="context"  # Nome esperado no prompt {context}.
        )

        # Encadeia recuperação + resposta para formar o pipeline RAG.
        self.__rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        self.guardrails = GuardrailsSecurity()  # Placeholder para futuras validações de entrada/saída.
        self._session_store[session_id] = InMemoryChatMessageHistory()
