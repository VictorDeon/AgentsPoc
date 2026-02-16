from utils import load_environment_variables, get_env_var
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from vetorial_db import results_by_chromadb
from langchain.schema import Document

load_environment_variables()


def main():
    GEMINI_API_KEY = get_env_var('GEMINI_API_KEY')

    # Instanciando um modelo de embeddings do google
    # permitindo transformar texto em vetores numéricos
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    store = LocalFileStore("./embeddings_cache")
    cached_embeddings: CacheBackedEmbeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        store,
        namespace="gemini-embedding-cache"
    )

    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash-lite",
    #     temperature=0,
    #     api_key=GEMINI_API_KEY
    # )

    # Verificar a dimensão dos embeddings
    test_embedding = cached_embeddings.embed_documents(["Olá, mundo!", "Testando cache de embeddings.", "aaaa", "Olá, mundo!"])
    print(f"Dimensão dos embeddings: {len(test_embedding[0])}")
    # No ambiente real seria preenchido por um pdf ou algum outro tipo de documento.
    company_documents = [
        Document(
            page_content="Política de férias: Funcionários têm direito a 30 dias de férias após 12 meses. A solicitação deve ser feita com 30 dias de antecedência.",
            metadata={"tipo": "política", "departamento": "RH", "ano": 2024, "id_doc": "doc001"}
        ),
        Document(
            page_content="Processo de reembolso de despesas: Envie a nota fiscal pelo portal financeiro. O reembolso ocorre em até 5 dias úteis.",
            metadata={"tipo": "processo", "departamento": "Financeiro", "ano": 2023, "id_doc": "doc002"}
        ),
        Document(
            page_content="Guia de TI: Para configurar a VPN, acesse vpn.nossaempresa.com e siga as instruções para seu sistema operacional.",
            metadata={"tipo": "tutorial", "departamento": "TI", "ano": 2024, "id_doc": "doc003"}
        ),
        Document(
            page_content="Código de Ética e Conduta: Valorizamos o respeito, a integridade e a colaboração. Casos de assédio não serão tolerados.",
            metadata={"tipo": "política", "departamento": "RH", "ano": 2022, "id_doc": "doc004"}
        )
    ]

    results_by_chromadb(company_documents, cached_embeddings)


if __name__ == "__main__":
    main()