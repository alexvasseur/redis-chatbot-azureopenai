import os
import streamlit as st
import uuid

from time import time

from dotenv import load_dotenv
print("Loading env before AppConfig")
load_dotenv("../.env", verbose=True)


from config import AppConfig

from redisvl.llmcache.semantic import SemanticCache

# we need those to fine tune vector dimension for openAI which is larger than default
import redis as redisclient
from redis.commands.search.field import VectorField
from redisvl.index import SearchIndex


from redisvl.vectorize.text import OpenAITextVectorizer
import tiktoken #required for OpenAI

from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_openai import AzureOpenAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.vectorstores import Redis
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents import AgentType
from langchain.agents import initialize_agent


# Load Global env


config = AppConfig()

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex


# Helpers

# helps to iterate on chunks of a large list
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

@st.cache_resource()
def configure_retriever(path):
    """Create the Redis Vector DB retrieval tool"""
    # Read documents
    docs = []
    for file in os.listdir(path):
        print(file, flush=True)
        loader = PyPDFLoader(os.path.join(path, file))
        docs.extend(loader.load())
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)
    # Create embeddings and store in vectordb
    # Implictly relies on env var see https://python.langchain.com/docs/integrations/text_embedding/azureopenai
    embeddings = AzureOpenAIEmbeddings(openai_api_version="2023-05-15", azure_deployment=config.OPENAI_AZURE_EMBEDDING_DEPLOYMENT)
    

    # Check if not already vectorized (currently at path level, not at path/file level)
    embeddingsDone = redisclient.Redis.from_url(config.REDIS_URL)
    embeddingsDoneForDoc = embeddingsDone.sismember("doc:pdf:path", path)
    if not embeddingsDoneForDoc:
        # Azure OpenAI limit inputs at 16 for now
        vectordb = None
        for splitN in chunker(splits, 16):
            if vectordb == None:
                vectordb = Redis.from_documents(
                    splitN, embeddings, redis_url=config.REDIS_URL, index_name="chatbot"
                )
            else:
                #foo = True
                vectordb.add_documents(splitN)
        embeddingsDone.sadd("doc:pdf:path", path)
    else:
        print("Found existing embeddings in 'doc:pdf:path' for "+ path, flush=True)
        vectordb = Redis.from_existing_index(
                embeddings,
                index_name="chatbot",
                redis_url=config.REDIS_URL,
                schema = None
        )

    # Define retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": config.RETRIEVE_TOP_K})
    tool = create_retriever_tool(retriever, "search_chevy_manual", "Searches and returns snippets from the Chevy Colorado 2022 car manual.")
    return tool


@st.cache_resource()
def configure_cache():
    """Set up the Redis LLMCache built with OpenAI Text Embeddings"""
    #llmcache_embeddings = OpenAITextVectorizer(
    #    model=config.OPENAI_AZURE_EMBEDDING_DEPLOYMENT,
    #    api_config={"api_key": config.OPENAI_API_KEY}
    #)

    # https://github.com/RedisVentures/redisvl/blob/10c192d2ab41a0aea330eea43266f12c8afbf2a3/redisvl/llmcache/semantic.py#L13
    # Defaults to 768 dimensions but we have 1536
    #index=SearchIndex("llmcache_index", "llmcache", [VectorField(
    #        "prompt_vector",
    #        "FLAT",
    #        {"DIM": 1536, "TYPE": "FLOAT32", "DISTANCE_METRIC": "COSINE"},
    #    )])
    #index.connect(config.REDIS_URL)

    return SemanticCache(
        redis_url=config.REDIS_URL,
        threshold=0.7, #config.LLMCACHE_THRESHOLD, # semantic similarity threshold
        #vectorizer=llmcache_embeddings,
    )


def configure_agent(chat_memory, tools: list):
    """Configure the conversational chat agent that can use the Redis vector db for RAG"""
    memory = ConversationBufferMemory(
        memory_key="chat_history", chat_memory=chat_memory, return_messages=True
    )
    chatLLM = AzureChatOpenAI(
        deployment_name=config.OPENAI_AZURE_LLM_DEPLOYMENT
        #openai_api_base=
        #openai_api_version="2023-05-15",
        #deployment_name=DEPLOYMENT_NAME,
        #openai_api_key=API_KEY,
        #openai_api_type="azure"
        #temperature=0.1,
    )
    PREFIX = """"You are a friendly AI assistant that can help you understand your Chevy 2022 Colorado vehicle based on the provided PDF car manual. Users can ask questions of your manual! You should not make anything up."""

    FORMAT_INSTRUCTIONS = """You have access to the following tools:

    {tools}

    Use the following format:

    '''
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    '''

    When you have gathered all the information required, respond to the user in a friendly manner.
    """

    SUFFIX = """

    Begin! Remember to give detailed, informative answers

    Previous conversation history:
    {chat_history}

    New question: {input}
    {agent_scratchpad}
    """
    return initialize_agent(
        tools,
        chatLLM,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        agent_kwargs={
            'prefix': PREFIX,
            'format_instructions': FORMAT_INSTRUCTIONS,
            'suffix': SUFFIX
        }
    )


class PrintRetrievalHandler(BaseCallbackHandler):
    """Callback to print retrieved source documents from Redis during RAG."""
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.container.write(f"**Document {idx} from {source}**")
            self.container.markdown(doc.page_content)


def generate_response(
    use_cache: bool,
    llmcache: SemanticCache,
    user_query: str,
    agent
) -> str:
    """Generate a response to the user's question after checking the cache (if enabled)."""
    t0 = time()
    if use_cache:
        if response := llmcache.check(user_query):
            print("Cache Response Time (secs)", time()-t0, flush=True)
            return response[0]
        else:
            print("No relevant cache entry found")

    retrieval_handler = PrintRetrievalHandler(st.container())
    response = agent.run(input=user_query, callbacks=[retrieval_handler])
    print("Full Response Time (secs)", time()-t0, flush=True)
    return response


def render():
    """Render the Streamlit chatbot user interface."""
    # Main Page
    st.set_page_config(page_title=config.PAGE_TITLE, page_icon=config.PAGE_ICON)
    st.title(config.PAGE_TITLE)

    # Setup LLMCache in Redis
    llmcache = configure_cache()

    # Setup Redis memory for conversation history
    msgs = RedisChatMessageHistory(
        session_id=st.session_state.session_id, url=config.REDIS_URL
    )

    # Sidebar
    with st.sidebar:
        st.image("https://redis.io/wp-content/uploads/2024/04/Logotype.svg?auto=webp&quality=90,75&width=240")
        st.markdown("""
**Vector Database** - *Making it easy to build Generative AI applications with Redis Enterprise.*
- Session store
- Vector store and indexing for GenAI embeddings, with hybrid search
- Semantic caching
- Retrieval Augmented Generation for LLM
- LLM memory             
and more!
                    
**Real time and Scalable for production** - *Perfectly suited for the massive performance and scalability requirements of AI applications.*
- Get started with open source, Redis Stack and a vibrant ecosystem
- Scale with Redis Enterprise
- In-memory architecture with full persistence and RAM/Flash auto-tiering 
- Highly available, clustered database and geo-cluster architecture
- On any cloud, Kubernetes or private cloud
- Fully managed with Redis Enterprise Cloud (GCP, AWS, Azure)                    
""")
        st.divider()
        use_cache = st.checkbox("Use LLM cache?")
        if st.button("Clear LLM cache"):
            llmcache.clear()
        if len(msgs.messages) == 0 or st.button("Clear message history"):
            msgs.clear()


    # Setup Redis vector db retrieval
    retriever = configure_retriever(config.DOCS_FOLDER)

    # Configure Agent
    agent = configure_agent(chat_memory=msgs, tools=[retriever])

    # Chat Interface
    avatars = {"human": "user", "ai": "assistant"}
    for msg in msgs.messages:
        if msg.type in avatars:
            with st.chat_message(avatars[msg.type]):
                st.markdown(msg.content)

    if user_query := st.chat_input(placeholder="Ask me anything about the 2022 Chevy Colorado!"):
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            print("Assistant loop - cache: " + str(use_cache))
            response = generate_response(use_cache, llmcache, user_query, agent)
            st.markdown(response)
            if use_cache:
                # TODO - should we cache responses that were used from the cache?
                llmcache.store(user_query, response)


if __name__ == "__main__":
    render()
