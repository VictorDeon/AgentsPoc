import os
from dotenv import load_dotenv


def load_environment_variables():
    """
    Carrega as variáveis de ambiente do arquivo .env
    """

    load_dotenv()


def get_env_var(key, default=None):
    """
    Retorna o valor da variável de ambiente ou o valor padrão
    """

    return os.getenv(key, default)
