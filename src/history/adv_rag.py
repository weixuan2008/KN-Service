from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Milvus

from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

folder_path = "../db"
question = "What can the Camunda do?"

# 1. Setup llm module
llm = Ollama(model="llama3.1",
             temperature=0.1,
             top_p=0.4,
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
             )

# 2. Setup embedding module
embedding = OllamaEmbeddings(model="nomic-embed-text")

# 3. Load pdf
loader = PDFPlumberLoader("../../data/pdf/camunda2.pdf")
docs = loader.load_and_split()

# 4. Setup text splitter and chunk pdf content
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)
chunks = text_splitter.split_documents(docs)

# 5.1 embed the chunks with server mode
# chroma_client = chromadb.HttpClient(host='127.0.0.1', port=8000, settings=Settings(allow_reset=True, anonymized_telemetry=False))
# print(chroma_client.heartbeat())
# collection = chroma_client.create_collection(name="sg-gpp3")
# chroma_db = Chroma(
#         collection_name="sg-gpp3",
#         embedding_function=embedding,
#         client=chroma_client,
#     )
# vector_store = chroma_db.from_documents(documents=chunks, embedding=embedding)

# 5.2Local mode
# vector_store = Chroma.from_documents(
#     documents=chunks, embedding=embedding, persist_directory=folder_path
# )

# 5.3 ElasticSearch
# es_store = ElasticsearchStore(
#     es_cloud_id="58a10608a6a644a0a4aa39efdab55c85:ZWFzdHVzMi5henVyZS5lbGFzdGljLWNsb3VkLmNvbTo0NDMkMDlmODNiZDcwZGI1NGJhODhhZWU3ZDYwNTdiM2QwOGMkMTc1ODg2Y2ZmMGFjNGU5MTg4MjY2MzhiMDQ0MGM0ZTM=",
#     es_api_key="MFVneXFaRUJxQ0QwX2JCSkVFMW06RTdBdjNiMFpURGlsaGNPM0xFMXhpZw==",
#     index_name="HK-DSC",
#     strategy=ElasticsearchStore.SparseVectorRetrievalStrategy(model_id=".elser_model_2"),
# ),
#
# vectorstore = ElasticsearchStore(
#     es_cloud_id="58a10608a6a644a0a4aa39efdab55c85:ZWFzdHVzMi5henVyZS5lbGFzdGljLWNsb3VkLmNvbTo0NDMkMDlmODNiZDcwZGI1NGJhODhhZWU3ZDYwNTdiM2QwOGMkMTc1ODg2Y2ZmMGFjNGU5MTg4MjY2MzhiMDQ0MGM0ZTM=",
#     es_api_key="MFVneXFaRUJxQ0QwX2JCSkVFMW06RTdBdjNiMFpURGlsaGNPM0xFMXhpZw==",
#     index_name="hk-dsc",
#     embedding=embedding,
# )
# vectorstore.add_texts(question)
# vector_store = vectorstore.from_documents(documents=chunks, embedding=embedding, index_name="hk-dsc",es_cloud_id="58a10608a6a644a0a4aa39efdab55c85:ZWFzdHVzMi5henVyZS5lbGFzdGljLWNsb3VkLmNvbTo0NDMkMDlmODNiZDcwZGI1NGJhODhhZWU3ZDYwNTdiM2QwOGMkMTc1ODg2Y2ZmMGFjNGU5MTg4MjY2MzhiMDQ0MGM0ZTM=",)

# 5.4 Milvus
# https://blog.csdn.net/2401_85378759/article/details/141160098
vector_store = Milvus.from_documents(
    documents=chunks,  # 设置保存的文档
    embedding=embedding,  # 设置 embedding model
    collection_name="hk_dsc_product",  # 设置 集合名称
    drop_old=True,
    connection_args={"host": "192.168.3.5", "port": "19530"},  # Milvus连接配置
)

# 6. Persist the database to disk
# vector_store.persist()

# 7. Retrieving the context from the DB using similarity search
retriever = vector_store.as_retriever(search_kwargs={'k': 3})
results = vector_store.similarity_search_with_score(question, k=3)

# 8. Define prompt template
template = f"""
You are an assistant for question-answering tasks.
Use the provided context only to answer the following question:

<context>
{context}
</context>

Question: {input}
"""

# 9. Create a prompt template
prompt = ChatPromptTemplate.from_template(template)

# 9. Create a chain
doc_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, doc_chain)

# 10. Invoke the Retrieval QA Chain
response = chain.invoke({"input": "what is Qlora?"})

# 11. Get the Answer only
response['answer']


https://academy.langchain.com/courses/intro-to-langgraph