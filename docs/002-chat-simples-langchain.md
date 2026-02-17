# Um Chat Simples Com LangChain

Na maioria das aulas vamos usar pelo menos um LLM para executar qualquer tarefa
que precisarmos.

Com o **LangChain**, você pode escolher qualquer LLM e provider de sua
preferência, seja pago, gratuito, local ou online.

---

## Ollama (local)

Quando estou testando alguma coisa com LLMs nos meus códigos, prefiro usar
modelos locais com **Ollama** (ou similares). Isso me permite fazer testes mais
complexos sem estourar o limite de tokens gratuitos ou gastar demais com APIs
pagas.

Se você também quiser usar Ollama, já fiz um vídeo sobre isso no canal:

- [Como Usar Ollama para LLMs no Seu Computador?](https://youtu.be/9Yz42WSISr4?si=aBVdWmfR8aQmtKD4)

---

## Google AI Studio (API gratuita)

Nem todo mundo tem hardware suficiente para rodar um modelo local. Geralmente
isso exige uma boa CPU/GPU e bastante espaço em disco e memória. Mesmo quando o
hardware dá conta, pode ser que o modelo não esteja otimizado para a sua máquina
e acabe rodando de forma tão lenta que inviabiliza os testes.

Nesses casos, a alternativa são **APIs externas** (gratuitas ou pagas).

No momento em que escrevo este texto, a **Google** oferece uma API Key gratuita
para desenvolvedores e entusiastas. Isso permite usar os modelos **Gemini**,
suficientes para nossos testes. Você pode gerar a chave no link:

- [Google AI Studio](https://aistudio.google.com/apikey)

---

## API paga

Se você não conseguir usar nenhuma das opções anteriores, sempre existe a
possibilidade de usar uma API paga. Geralmente, o custo inicial não é alto (no
momento em torno de 5 dólares de créditos mínimos).

Basta criar uma conta em um provider como Google, Anthropic, OpenAI ou outro,
adicionar créditos e obter sua **API Key**.

Links diretos para geração de chave:

- [Google AI Studio](https://aistudio.google.com/apikey)
- [OpenAI API Key](https://platform.openai.com/api-keys)
- [Anthropic API Key](https://console.anthropic.com/settings/keys)

---

## Variáveis de ambiente e o arquivo `.env`

Se optar por uma API externa (paga ou gratuita), é importante carregar suas
variáveis de ambiente de forma **segura**.

Crie um arquivo chamado `.env` na raiz do seu projeto com o seguinte conteúdo:

```sh
ANTHROPIC_API_KEY="VALOR"
OPENAI_API_KEY="VALOR"
GOOGLE_API_KEY="VALOR"
```

Para carregar esse arquivo, você pode usar o pacote `python-dotenv` ou deixar o
`uv` cuidar disso automaticamente:

```bash
uv run --env-file=".env" caminho/do/arquivo.py
```

Depois de carregar o `.env`, teste se as variáveis estão acessíveis com:

```python
import os

print(f"{os.getenv('GOOGLE_API_KEY', 'Não configurado')=}")
print(f"{os.getenv('ANTHROPIC_API_KEY', 'Não configurado')=}")
print(f"{os.getenv('OPENAI_API_KEY', 'Não configurado')=}")
```

Se os valores aparecerem corretamente, está tudo certo.

**IMPORTANTE:** nunca versionar o arquivo `.env` nem expor suas API Keys no
GitHub. Adicione o `.env` ao `.gitignore` do projeto para evitar problemas de
segurança.
