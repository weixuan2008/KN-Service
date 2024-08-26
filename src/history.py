from flask import Flask, request
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = Flask(__name__)

chat_history = []
folder_path = "../db"

llm = Ollama(model="llama3.1",
             temperature=0.1,
             top_p=0.4,
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
             )

embedding = OllamaEmbeddings(model="nomic-embed-text")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """<s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the 
    provided information say so. [/INST] </s> [INST] {input} Context: {context} Answer: [/INST]"""
)


@app.route("/health", methods=["GET"])
def health():
    print("Get /health called")
    response_answer = {"answer": "I am healthy."}
    return response_answer


@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    response = llm.invoke(query)

    print(response)

    response_answer = {"answer": response}
    return response_answer


@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )

    prompt_template = """基于以下已知内容，简洁和专业的来回答用户的问题。
                                                如果无法从中得到答案，清说"根据已知内容无法回答该问题"
                                                答案请使用中文。
                                                已知内容:
                                                {context}
                                                问题:
                                                {question}"""

    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=prompt
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    # chain = create_retrieval_chain(retriever, document_chain)

    retrieval_chain = create_retrieval_chain(
        # retriever,
        history_aware_retriever,
        document_chain,
    )

    # result = chain.invoke({"input": query})
    result = retrieval_chain.invoke({"question": query})
    print(result["answer"])
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=result["answer"]))

    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )

    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer


@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = "../data/pdf/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )

    vector_store.persist()

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return response


def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    start_app()


@app.route("/")
def hello():
    return "Hello, Flask!"


if __name__ == "__main__":
    app.run()
