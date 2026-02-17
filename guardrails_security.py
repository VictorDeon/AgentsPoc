from __future__ import annotations

import re
from dataclasses import dataclass, field
from langchain_core.messages import SystemMessage


@dataclass
class GuardrailsSecurity:
    """
    Valida entradas e saídas do pipeline para mitigar riscos.
    """

    max_input_chars: int = 2000
    max_output_chars: int = 4000
    blocked_phrases: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.blocked_phrases:
            self.blocked_phrases = [
                "ignore as instruções",
                "ignore instruções",
                "ignore the instructions",
                "disregard previous instructions",
                "reveal the system prompt",
                "mostre o prompt do sistema",
                "prompt do sistema",
                "system prompt",
                "bypass safety",
                "jailbreak",
                "vaze as chaves",
                "exfiltrate",
                "dump .env",
                "leia o arquivo .env",
                "mostre as chaves",
                "mostre a chave",
                "api key",
                "chave de api",
                "chave de acesso",
                "apikey",
                "token",
                "senha",
                "segredo",
                "secrets",
            ]

        self._blocked_regex = [
            re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"),  # Google API key
            re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),  # OpenAI-like key
            re.compile(r"(?i)gemini_api_key"),
            re.compile(r"(?i)api_key"),
            re.compile(r"(?i)senha"),
            re.compile(r"(?i)token"),
            re.compile(r"(?i)secret"),
            re.compile(r"(?i)\.env"),
        ]

    def validate_input(self, text: str) -> str:
        """
        Valida texto de entrada do usuário.

        Raises:
            ValueError: quando o texto é inseguro.
        """

        if text is None or not str(text).strip():
            raise ValueError("Entrada vazia não é permitida.")

        normalized = str(text).strip()
        if len(normalized) > self.max_input_chars:
            raise ValueError("Entrada muito longa.")

        lowered = normalized.lower()
        for phrase in self.blocked_phrases:
            if phrase in lowered:
                raise ValueError("Entrada potencialmente insegura.")

        return normalized

    def validate_output(self, content: dict) -> str:
        """
        Valida texto de saída do modelo.
        """

        messages = content.get("messages", [])

        if not messages:
            raise ValueError("Saída vazia não é permitida.")

        last_message: SystemMessage = messages[-1]
        message_id = last_message.id
        model_name = last_message.response_metadata.get("model_name", "desconhecido")
        model_provider = last_message.response_metadata.get("model_provider", "desconhecido")
        metadata = last_message.response_metadata

        print(f"Validando saída do modelo '{model_name}' (ID: {message_id}, Provider: {model_provider}) com metadata: {metadata}")
        text = last_message.content

        normalized = str(text).strip()
        if len(normalized) > self.max_output_chars:
            raise ValueError("Saída muito longa.")

        for pattern in self._blocked_regex:
            if pattern.search(normalized):
                print(f"Saída bloqueada por regex: {normalized}")
                raise ValueError("Saída contém possível informação sensível.")

        return normalized
