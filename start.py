import uvicorn
import logging


if __name__ == "__main__":
    print("Iniciando API de desenvolvimento")
    uvicorn.run("api.main:app", host="0.0.0.0", port=3100, workers=1, reload=True, log_level=logging.INFO)
