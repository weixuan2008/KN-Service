from langchain_community.llms.ollama import Ollama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI

# 1. Setup llm module
llm = Ollama(model="llama3.1",
             temperature=0.1,
             top_p=0.4,
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
             )
# 2. Setup embedding module
embedding = OllamaEmbeddings(model="nomic-embed-text")

# messages = [
#     SystemMessage(content="Translate the following from English into Italian"),
#     HumanMessage(content="hi!"),
# ]
#
# parser = StrOutputParser()
# chain = llm | parser
# chain.invoke(messages)

# 3. Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

result = prompt_template.invoke({"language": "italian", "text": "hi"})
ret = result.to_messages()

# 4. Create parser
parser = StrOutputParser()

# 5. Create chain
chain = prompt_template | llm | parser

chain.invoke({"language": "italian", "text": "hi"})

idd = 0
