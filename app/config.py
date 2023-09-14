import os


class AppConfig:
    DOCS_FOLDER=os.environ["DOCS_FOLDER"]
    REDIS_URL=os.environ["REDIS_URL"]
    
    CHUNK_SIZE=int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP=int(os.getenv("CHUNK_OVERLAP", 10))
    PAGE_TITLE=os.getenv("PAGE_TITLE", "📃 Chat Your PDF")
    PAGE_ICON=os.getenv("PAGE_ICON", "📃")
    RETRIEVE_TOP_K=int(os.getenv("RETRIEVE_TOP_K", 5))
    LLMCACHE_THRESHOLD=float(os.getenv("LLMCACHE_THRESHOLD", 0.75))

    OPENAI_AZURE_EMBEDDING_DEPLOYMENT=os.environ["OPENAI_AZURE_EMBEDDING_DEPLOYMENT"]
    OPENAI_AZURE_LLM_DEPLOYMENT=os.environ["OPENAI_AZURE_LLM_DEPLOYMENT"]
    OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
