from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.graph.state import RunnableConfig
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from enum import Enum
from utils import checkpointer
from dtos import MainContext, QuestionInputDTO
from typing import TypedDict, Annotated, Sequence
from rich import print


@tool
def multiply_subtool(a: float, b: float) -> float:
    """
    Multiplica dois números.

    Args:
        a: O primeiro número.
        b: O segundo número.

    Returns:
        O resultado da multiplicação de a e b.
    """

    return a * b


@tool
def add_subtool(a: float, b: float) -> float:
    """
    Soma dois números.

    Args:
        a: O primeiro número.
        b: O segundo número.

    Returns:
        O resultado da soma de a e b.
    """

    return a + b


@tool
def subtract_subtool(a: float, b: float) -> float:
    """
    Subtrai dois números.

    Args:
        a: O primeiro número.
        b: O segundo número.

    Returns:
        O resultado da subtração de a e b.
    """

    return a - b


@tool
def divide_subtool(a: float, b: float) -> float:
    """
    Divide dois números.

    Args:
        a: O primeiro número (dividendo).
        b: O segundo número (divisor).

    Returns:
        O resultado da divisão de a por b, ou uma mensagem de erro se b for zero.
    """

    if b == 0:
        return "Divisão por zero não é permitida."

    return a / b


class GraphType(Enum):
    CALL_LLM = "call_llm"
    TOOL_NODE = "tools"


class ToolState(TypedDict):
    # Usamos Annotated para marcar que esses campos são mensagens criadas pelo add_messages que devem ser
    # adicionadas ao contexto da conversa. (SystemMessage, HumanMessage, AIMessage, etc.)
    messages: Annotated[Sequence[BaseMessage], add_messages]


TOOLS = [multiply_subtool, add_subtool, subtract_subtool, divide_subtool]


def call_llm(state: ToolState) -> ToolState:
    """
    Node que carrega a llm com as ferramentas matemáticas.
    """

    llm = init_chat_model(model="google_genai:gemini-2.5-flash-lite")
    llm_with_tools = llm.bind_tools(TOOLS)
    llm_result = llm_with_tools.invoke(state["messages"])
    return ToolState(messages=[llm_result])


tool_node = ToolNode(tools=TOOLS)


@tool(args_schema=QuestionInputDTO)
def graph_tool(question: str, runtime: ToolRuntime[MainContext]) -> str:
    """
    Utilize esta ferramenta SEMPRE que o usuário pedir alguma conta matemática básica como
    soma, multiplicação, divisão ou subtração.

    Args:
        question: A pergunta do usuário relacionada a operações matemáticas básicas.
        runtime: O contexto de execução da ferramenta, fornecido pelo agente.

    Returns:
        A resposta gerada pelo modelo de linguagem após processar a pergunta do usuário.

    Ferramentas Disponíveis:
        - multiply_subtool: Multiplica dois números.
        - add_subtool: Soma dois números.
        - subtract_subtool: Subtrai dois números.
        - divide_subtool: Divide dois números.
    """

    print(f"Entrei na ferramenta 'graph_tool' com a pergunta: \"{question}\"")

    context = runtime.context

    builder = StateGraph(ToolState, context_schema=MainContext)

    # Adicionando os nós
    builder.add_node(GraphType.CALL_LLM.value, call_llm)
    builder.add_node(GraphType.TOOL_NODE.value, tool_node)

    # Adicionando as arestas
    # __start__ -> call_llm -> (condicional) -> tool_node -> call_llm -> (condicional) -> __end__
    builder.add_edge(START, GraphType.CALL_LLM.value)
    # Do call_llm ele roda o tools_condition que valida em qual dos casos na lista ele vai seguir.
    builder.add_conditional_edges(GraphType.CALL_LLM.value, tools_condition, [GraphType.TOOL_NODE.value, END])
    # Se optar a conditional seguir o tool_node, ele vai rodar a ferramenta e voltar ao call_llm para rodar novamente o loop
    # até o condition cair em __end__
    builder.add_edge(GraphType.TOOL_NODE.value, GraphType.CALL_LLM.value)

    graph = builder.compile(checkpointer=checkpointer)

    # Usar thread_id isolado para evitar conflito de estrutura com o agente principal
    config = RunnableConfig(
        configurable={"thread_id": f"graph_tool_{context.session_id}"}
    )

    result = graph.invoke(
        {"messages": [
            SystemMessage(content="""
                Você é um assistente matemático. Após executar as ferramentas matemáticas,
                SEMPRE responda com o resultado em linguagem natural, dizendo apenas o número
                resultado sem explicações adicionais.

                Ferramentas Disponíveis:
                    - multiply_subtool: Multiplica dois números.
                    - add_subtool: Soma dois números.
                    - subtract_subtool: Subtrai dois números.
                    - divide_subtool: Divide dois números.
            """),
            HumanMessage(question)
        ]},
        config=config,
        context=context
    )

    last_message = result["messages"][-1]
    answer = last_message.content if result["messages"] else "Desculpe, não consegui gerar uma resposta."

    return answer