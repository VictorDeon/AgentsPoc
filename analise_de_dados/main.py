from utils import load_environment_variables, get_env_var, get_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.tools import PythonAstREPLTool
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_groq import ChatGroq
import matplotlib.pyplot as plt
from functools import partial
import seaborn as sns
import pandas as pd

load_environment_variables()


def dataframe_informations(question: str, df: pd.DataFrame, llm: ChatGroq) -> str:
    """
    Utilize esta ferramenta sempre que o usuário solicitar informações gerais
    sobre o DataFrame, incluindo número de colunas e linhas, nomes das colunas,
    e seus tipos de dados, contagem de dados nulos e duplicados para dar um
    panorama geral sobre o arquivo.
    """

    shape = df.shape
    columns = df.dtypes
    nulls = df.isnull().sum()
    nulls_str = df.apply(lambda col: col[~col.isna()].astype(str).str.strip().str.lower().eq("nan").sum())
    duplicates = df.duplicated().sum()

    prompt = get_prompt('exploratoria.prompt.md')

    response_template = PromptTemplate(
        template=prompt,
        input_variables=["question", "shape", "columns", "nulls", "nulls_str", "duplicates"]
    )

    chain = response_template | llm | StrOutputParser()

    response = chain.invoke({
        "question": question,
        "shape": shape,
        "columns": columns,
        "nulls": nulls,
        "nulls_str": nulls_str,
        "duplicates": duplicates
    })

    return response


def statistical_summary(question: str, df: pd.DataFrame, llm: ChatGroq) -> str:
    """
    Utilize esta ferramenta sempre que o usuário solicitar um resumo estatístico
    sobre as colunas numéricas do DataFrame, incluindo medidas como média,
    mediana, desvio padrão, valores mínimos e máximos, e contagem de valores
    únicos para colunas categóricas.
    """

    descritive_statistics = df.describe(include='number').transpose().to_string()

    prompt = get_prompt('estatistica.prompt.md')

    response_template = PromptTemplate(
        template=prompt,
        input_variables=["question", "summary"]
    )

    chain = response_template | llm | StrOutputParser()

    response = chain.invoke({
        "question": question,
        "summary": descritive_statistics
    })

    return response


def graph_generator(question: str, df: pd.DataFrame, llm: ChatGroq) -> plt.Figure:
    """
    Utilize esta ferramenta sempre que o usuário solicitar um gráfico a partir
    de um DataFrame pandas (`df`) com base em uma instrução do usuário. A instrução
    pode conter pedidos como:
    - "Crie um gráfico da média de tempo de entrega por clima."
    - "Plote a distribuição do tempo de entrega."
    - "Plote a relação entre a classificação dos agentes e o tempo de entrega"

    Palavras-chave comuns que indicam o uso desta ferramenta incluem:
    - "crie um gráfico"
    - "plote"
    - "visualize"
    - "faça um gráfico de"
    - "mostre a distribuição de"
    - "represente graficamente"

    Entre outros pedidos e palavras-chave que indicam a necessidade de gerar um gráfico a partir dos dados do DataFrame.
    """

    columns = [f"- {col}: ({dtype})" for col, dtype in df.dtypes.items()]
    samples = df.head(20).to_dict(orient='records')

    prompt = get_prompt('visual.prompt.md')

    response_template = PromptTemplate(
        template=prompt,
        input_variables=["question", "columns", "sample"]
    )

    chain = response_template | llm | StrOutputParser()

    response_code = chain.invoke({
        "question": question,
        "columns": "\n".join(columns),
        "sample": samples
    })

    clean_code = response_code.replace("```python", "").replace("```", "").strip()

    exec_globals = {"df": df, "plt": plt, "sns": sns}
    exec_locals = {}
    exec(clean_code, exec_globals, exec_locals)

    fig = plt.gcf()  # Get the current figure after executing the code

    return fig


def main():
    GROQ_API_KEY = get_env_var('GROQ_API_KEY')

    df = pd.read_csv('./analise_de_dados/assets/dados_entregas.csv')

    df_head = df.head().to_markdown()
    prompt_react = get_prompt('react.prompt.md')

    prompt_react_template = PromptTemplate(
        template=prompt_react,
        input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
        partial_variables={"df_head": df_head}
    )

    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model='llama-3.3-70b-versatile'
    )

    dataframe_informations_partial_tool = partial(dataframe_informations, df=df, llm=llm)
    dataframe_informations_tool = Tool(
        name="DataFrame Informations",
        func=lambda question: dataframe_informations_partial_tool(question),
        return_direct=True,
        description=""""
            Utilize esta ferramenta sempre que o usuário solicitar
            informações gerais sobre o dataframe, incluindo número
            de colunas e linhas, nomes das colunas e seus tipos
            de dados, contagem de dados nulos e duplicados para
            dar um panorama geral sobre o arquivo.
        """
    )

    statistical_summary_partial_tool = partial(statistical_summary, df=df, llm=llm)
    statistical_summary_tool = Tool(
        name="Statistical Summary",
        func=lambda question: statistical_summary_partial_tool(question),
        return_direct=True,
        description=""""
            Utilize esta ferramenta sempre que o usuário solicitar um
            resumo estatístico completo e descritivo da base de dados,
            incluindo várias estatísticas (média, desvio padrão,
            mínimo, máximo etc.). Não utilize esta ferramenta para
            calcular uma única métrica como 'qual é a média de X'
            ou qual a correlação das variáveis'. Nesses casos,
            utilize a ferramenta python_executor.
        """
    )

    graph_partial_tool = partial(graph_generator, df=df, llm=llm)
    graph_tool = Tool(
        name="Graph Generator",
        func=lambda question: graph_partial_tool(question),
        return_direct=True,
        description=""""
            Utilize esta ferramenta sempre que o usuário solicitar um
            gráfico a partir de um DataFrame pandas (df) com base em
            uma instrução do usuário. A instrução pode conter
            pedidos como: 'Crie um gráfico da média de tempo de entrega por clima',
            'Plote a distribuição do tempo de entrega' ou
            'Plote a relação entre a classificação dos agentes e o tempo de entrega'.
            Palavras-chave comuns que indicam o uso desta ferramenta incluem:
            'crie um gráfico', 'plote', 'visualize', 'faça um gráfico de',
            'mostre a distribuição', 'represente graficamente', entre outros.
        """
    )

    python_executor_tool = Tool(
        name="Python Executor",
        func=PythonAstREPLTool(locals={"df": df}),
        description=""""
            Utilize esta ferramenta sempre que o usuário solicitar cálculos,
            consultas ou transformações específicas usando Python diretamente
            sobre o DataFrame df. Exemplos de uso incluem: 'Qual é a média da coluna X?',
            'Quais são os valores únicos da coluna Y?' ou 'Qual a correlação entre A e B?'.
            Evite utilizar esta ferramenta para solicitações mais amplas ou descritivas,
            como informações gerais sobre o DataFrame, resumos estatísticos completos
            ou geração de gráficos — nesses casos, use as ferramentas apropriadas.
        """
    )

    tools = [dataframe_informations_tool, statistical_summary_tool, graph_tool, python_executor_tool]

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt_react_template
    )

    maestro = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        max_execution_time=60
    )

    response = maestro.invoke({"input": "Qual é a média do tempo de entrega?"})

    print(response["output"])


if __name__ == "__main__":
    main()
