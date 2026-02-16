from utils import load_environment_variables, get_env_var
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import time

load_environment_variables()


def compare_models():
    """
    Faz a comparação entre os modelos de embeddings do Google e do Hugging Face usando a métrica de similaridade de cosseno.
    """

    textos_teste = [
        "Qual é a política de férias da nossa empresa?",
        "Preciso de um relatório de despesas de viagem.",
        "Como configuro o acesso à rede privada virtual (VPN)?",
        "Onde encontro o código de conduta da organização?",
        "Quero entender o processo de avaliação de performance."
    ]

    question = "Quero tirar uns dias de folga do trabalho, como faço isso?"

    GEMINI_API_KEY = get_env_var('GEMINI_API_KEY')
    embedding_model_gemini = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    embedding_model_lm = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embedding_model_bge = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

    # Vamos entender qual dos 3 modelos melhor performa em responder a pergunta abaixo
    # Quero tirar uns dias de folga do trabalho, como faço isso?

    embeddings = embedding_model_gemini.embed_query(question)
    start_time = time.time()
    embeddings_gemini = embedding_model_gemini.embed_documents(textos_teste)
    end_time = time.time() - start_time
    print(f"Tempo de geração dos embeddings do Gemini: {end_time:.4f} segundos com dimensão {len(embeddings)}")
    similarities = cosine_similarity([embeddings], embeddings_gemini)[0]
    doc_and_similarities = sorted(
        zip(textos_teste, similarities), key=lambda x: x[1], reverse=True
    )
    print("--- Ranking para o modelo Gemini ---")
    for i, (doc, sim) in enumerate(doc_and_similarities[:3], 1):
        print(f"{i}. (Score: {sim:.3f}) {doc}")
    print()

    embeddings = embedding_model_lm.embed_query(question)
    start_time = time.time()
    embeddings_lm = embedding_model_lm.embed_documents(textos_teste)
    end_time = time.time() - start_time
    print(f"Tempo de geração dos embeddings do Hugging Face (all-MiniLM-L6-v2): {end_time:.4f} segundos com dimensão {len(embeddings)}")
    similarities = cosine_similarity([embeddings], embeddings_lm)[0]
    doc_and_similarities = sorted(
        zip(textos_teste, similarities), key=lambda x: x[1], reverse=True
    )
    print("--- Ranking para o modelo Hugging Face (all-MiniLM-L6-v2) ---")
    for i, (doc, sim) in enumerate(doc_and_similarities[:3], 1):
        print(f"{i}. (Score: {sim:.3f}) {doc}")
    print()

    embeddings = embedding_model_bge.embed_query(question)
    start_time = time.time()
    embeddings_bge = embedding_model_bge.embed_documents(textos_teste)
    end_time = time.time() - start_time
    print(f"Tempo de geração dos embeddings do Hugging Face (BAAI/bge-large-en-v1.5): {end_time:.4f} segundos com dimensão {len(embeddings)}")
    similarities = cosine_similarity([embeddings], embeddings_bge)[0]
    doc_and_similarities = sorted(
        zip(textos_teste, similarities), key=lambda x: x[1], reverse=True
    )
    print("--- Ranking para o modelo Hugging Face (BAAI/bge-large-en-v1.5) ---")
    for i, (doc, sim) in enumerate(doc_and_similarities[:3], 1):
        print(f"{i}. (Score: {sim:.3f}) {doc}")
    print()