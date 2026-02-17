from chat import ChatbotSingleton


def main(question: str):
    """
    Orquestra todo o fluxo de RAG, do ETL à resposta da pergunta.
    """

    # Usa uma sessão fixa para CLI, pode ser dinâmica em API.
    chat = ChatbotSingleton(session_id="default")

    # Pergunta de exemplo (pode ser substituída por entrada do usuário).
    try:
        question = chat.guardrails.validate_input(question)
        # Executa o pipeline RAG com uma sessão fixa ("default").
        response = chat.chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": "default"}}
        )
        # Resposta final do modelo com base nos documentos recuperados.
        safe_answer = chat.guardrails.validate_output(response['answer'])
        print(f"Resposta: {safe_answer}")
        # Documentos usados na resposta para auditoria/inspeção.
        # print(f"\nDocumentos utilizados: {len(response['context'])}")
    except Exception as e:
        # Captura qualquer erro de execução para facilitar debug.
        print(f"Erro ao processar a pergunta: {e}")


if __name__ == "__main__":
    """
    Exemplo de perguntas:
    - "O que é RAG"
    - "para que ela serve?"
    """

    # Ponto de entrada para execução via CLI.
    print("Chatbot iniciado. Digite sua pergunta ou 'sair' para encerrar.")
    while True:
        question = input("\nPergunta: ").strip()
        if not question:
            print("Por favor, digite uma pergunta válida.")
            continue
        if question.lower() in {"sair", "exit", "quit"}:
            print("Encerrando o chatbot.")
            break

        main(question)