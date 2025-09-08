import subprocess
from core.constants import OLLAMA_MODEL, OLLAMA_LIPREAD_PROMPT

class OllamaClient:
    """
    OllamaClient class
    ---------------------
    Connection handler for the Ollama Model
    used to process the after effects of the lipreading model
    """
    def __init__(self):
        self.model = OLLAMA_MODEL

    def interpret(self, text):
        proc = subprocess.Popen(
            ["ollama", "run", self.model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        response, _ = proc.communicate(input=f"{OLLAMA_LIPREAD_PROMPT}\n\nContext: {text}")
        return response.strip()