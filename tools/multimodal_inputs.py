from utils import get_env_var
from typing import Literal
from langchain.tools import tool, ToolRuntime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dtos import MainContext, AttachmentInputDTO


@tool(args_schema=AttachmentInputDTO)
def multimodal_inputs_tool(
    question: str,
    attachment_type: Literal["image", "video"],
    attachment_url: str,
    runtime: ToolRuntime[MainContext]
) -> str:
    """
    Utilize esta ferramenta para responder perguntas que tenha anexos de imagem ou video, utilize
    esses anexos para extrair informações relevantes e responder a pergunta do usuário. Exemplos de perguntas incluem:
    - "Qual é o conteúdo do vídeo anexo?"
    - "O que está escrito na imagem anexa?"
    - "Quais são os objetos presentes na imagem anexa?"
    - "O vídeo anexo é relevante para a pergunta?"
    - "Quais insights podem ser extraídos do vídeo anexo em relação à pergunta?"
    - "A imagem anexa contém informações que respondem à pergunta?"
    - "Quais são os principais pontos discutidos no vídeo anexo?"
    - "A imagem anexa tem alguma relação com a pergunta?"
    - "O que a imagem anexa revela sobre o assunto da pergunta?"

    Args:
        question: A pergunta do usuário relacionada ao anexo.
        attachment_type: O tipo do anexo, pode ser 'image' ou 'video'.
        attachment_url: A URL do anexo, que pode ser uma imagem ou um vídeo.
        runtime: O contexto de execução da ferramenta, fornecido pelo agente.
    """

    print(f"Entrei na ferramenta 'multimodal_inputs_tool' com a pergunta: \"{question}\"")

    context = runtime.context

    if attachment_type != "image":
        return "No momento, esta ferramenta suporta apenas anexos de imagem."

    GEMINI_API_KEY = get_env_var('GEMINI_API_KEY')

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        api_key=GEMINI_API_KEY
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": attachment_url}
        ]
    )

    response = llm.invoke(
        [message],
        config={"configurable": {"thread_id": context.session_id}}
    )

    return response.content