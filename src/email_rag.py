import chromadb
from chromadb import Settings
from flask import Flask, request
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter

folder_path = "../db"
# question = "Explain how the YOLO method works"
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
loader = PDFPlumberLoader("../data/pdf/camunda2.pdf")
docs = loader.load_and_split()

# 4. Setup text splitter and chunk pdf content
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)
chunks = text_splitter.split_documents(docs)

# 5. embed the chunks with server mode
chroma_client = chromadb.HttpClient(host='127.0.0.1', port=8000, settings=Settings(allow_reset=True, anonymized_telemetry=False))
print(chroma_client.heartbeat())
# collection = chroma_client.create_collection(name="sg-gpp3")
chroma_db = Chroma(
        collection_name="sg-gpp3",
        embedding_function=embedding,
        client=chroma_client,
    )
vector_store = chroma_db.from_documents(documents=chunks, embedding=embedding)

# Local mode
# vector_store = Chroma.from_documents(
#     documents=chunks, embedding=embedding, persist_directory=folder_path
# )

# 6. Persist the database to disk
# vector_store.persist()

# 7. Retrieving the context from the DB using similarity search
retriever = vector_store.as_retriever(search_kwargs={'k': 3})
results = vector_store.similarity_search_with_score(question, k=3)

loop = 0
for doc, score in results:
    ++loop
    print("---------------------------doc " + str(loop) + "-------------------------------")
    print(doc)
    print("---------------------------score " + str(loop) + "-------------------------------")
    print(score)

# 8. Config mail template
prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
prompt = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question'])

# 9. Instantiate the Retrieval Question Answering Chain
qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=retriever,
                                 return_source_documents=True,
                                 chain_type_kwargs={"prompt": prompt})

# 10. Invoke the Retrieval QA Chain
response = qa.invoke({"query": question})
print(response.values())
print(response['result'])
