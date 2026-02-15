from utils import load_environment_variables, get_env_var

load_environment_variables()


def main():
    # Exemplo de uso
    api_key = get_env_var('GROQ_API_KEY')
    print(f"API Key: {api_key}")
    ...


if __name__ == "__main__":
    main()
