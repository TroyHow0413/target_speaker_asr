"""Web UI + OpenAI-compatible /v1 API on one port, one set of loaded models.

    python run_server_openai.py [--config config.yaml] [--port 8001] [--profile alice] [--ssl]
"""
from tsasr.server import main

if __name__ == "__main__":
    main(openai=True)
