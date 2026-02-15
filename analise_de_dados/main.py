from utils import load_environment_variables, get_env_var, get_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_groq import ChatGroq
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

load_environment_variables()


@tool
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


@tool
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


@tool
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
    samples = df.head(3).to_dict(orient='records')

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

    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model='llama-3.3-70b-versatile'
    )

    # dataframe_informations_tool = dataframe_informations.invoke({
    #     "question": "Quais são as informações gerais sobre o DataFrame?",
    #     "df": df,
    #     "llm": llm
    # })

    # print(dataframe_informations_tool)

    # descritive_summary_tool = statistical_summary.invoke({
    #     "question": "Quais as estatísticas descritivas dos dados?",
    #     "df": df,
    #     "llm": llm
    # })

    # print(descritive_summary_tool)

    graph_response: plt.Figure = graph_generator.invoke({
        "question": "Crie um gráfico da média de tempo de entrega por clima. Ordene do maior para o menor valor.",
        "df": df,
        "llm": llm
    })

    graph_response.show()


if __name__ == "__main__":
    main()
