# Usa uma imagem Python leve
FROM python:3.11-slim

# Define diretório de trabalho
WORKDIR /app

# Copia e instala dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código fonte
COPY server.py .

# Expõe a porta que definimos no uvicorn
EXPOSE 8000

# Comando para iniciar o servidor
CMD ["python", "server.py"]
