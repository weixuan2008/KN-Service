import sys
sys.path.append('D:/BaiduSyncdisk/LLM应用培训/数据治理实验')
env_path = 'D:/BaiduSyncdisk/LLM应用培训/数据治理实验/.env'
import openai 
import os
from dotenv import load_dotenv
load_dotenv (dotenv_path=env_path)

#LLM with local ChatGLM3
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
llm= ChatOpenAI(
    model_name="chatglm3-6b",
    streaming=True, 
    callbacks=[StreamingStdOutCallbackHandler()],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base=os.environ["OPENAI_API_BASE"]                
)

#清华智谱GLM4
llmglm4 = ChatOpenAI(
    temperature=0.95,
    model="glm-4",
    openai_api_key=os.environ["GLM4_API_KEY"],
    openai_api_base=os.environ["GLM4_API_BASE"]
)

#月之暗面
llmkimi = ChatOpenAI(
    temperature=0.3,
    model="moonshot-v1-8k",
    openai_api_key=os.environ["MOONSHOT_API_KEY"],
    openai_api_base=os.environ["MOONSHOT_API_BASE"]
)

#LLM with Baidu Wenxin
from langchain_wenxin.chat_models import ChatWenxin
from langchain.schema.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
llmwx = ChatWenxin(temperature=0.1,streaming=True, callbacks=[StreamingStdOutCallbackHandler()],)

from langchain.prompts import PromptTemplate
header1_template=PromptTemplate(
    input_variables=["topic"],
    template="写一个虚构小说的标题，内容是关于'{topic}'"
)

header2_template=PromptTemplate(
    input_variables=["title","topic"],
    template="参考如下虚构小说的背景，为标题是'{title}'的文章写目录的章节标题。\n虚构小说的背景：\n{topic}\n\n目录:",
)
count_template=PromptTemplate(
    input_variables=["toc"],
    template="下面有几个章节标题：\n{toc}\n\n标题的数目为(只要数字):",
)
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories.streamlit import StreamlitChatMessageHistory

title_memory = ConversationBufferMemory(memory_key="history", input_key="topic")
section_memory = ConversationBufferMemory(memory_key="history", input_key="title")
count_memory = ConversationBufferMemory(memory_key="history", input_key="toc")
content_memory = ConversationBufferMemory(memory_key="history", input_key="no")

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
content_template = """结合这个虚拟小说的背景和目录，根据选择的章节名称用中文完成段落内容的编写。
背景:
{topic}
目录:
{toc}
选择的章节:第{no}章
"""

prompt = PromptTemplate(input_variables=["topic","toc", "no"], template=content_template)

llm_title_chain = LLMChain(llm=llm, prompt=header1_template, memory=title_memory,verbose=True,output_key="title")
llm_section_chain = LLMChain(llm=llm, prompt=header2_template, memory=section_memory,verbose=True,output_key="toc")
llm_count_chain = LLMChain(llm=llm, prompt=count_template, memory=count_memory,verbose=True,output_key="no")
from langchain.chains.sequential import SequentialChain

llm_content_chain = LLMChain(llm=llm, prompt=prompt, memory=content_memory,verbose=True,output_key="content")

from langchain_core.globals import set_debug
set_debug(True)

import streamlit as st
import re
from thinking import StreamlitCallbackHandler
st.chat_message("assistant").write("请提供简要故事背景，包括人、事、物。以及故事梗概。AI帮你实现各章节的撰写。")
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            
            seqChain = SequentialChain(chains=[llm_title_chain, llm_section_chain,llm_count_chain],
                input_variables=["topic"],
                # Here we return multiple variables
                output_variables=["title", "toc","no"],
                verbose=True,
                callbacks=[st_cb]
            )
            a=seqChain({"topic":prompt})
            print(a)
            print(a["toc"])
            
            msgList=[]
            msgList.append(a["title"])
            msgList.append(a["toc"])

            charter_digi=re.findall(r'\d+',a["no"])
            charter_no=charter_digi[0]
            
            for i in range(int(charter_no)):
                st_cb1 = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True,collapse_completed_thoughts=True)
                c = llm_content_chain.run({"no":i+1,"toc":a["toc"],"title":a["title"],"topic":prompt},callbacks=[st_cb1])
                msgList.append(c)
            
            for msg in msgList:
                st.markdown(msg)
            
    with open('story_result.txt', 'w') as file:
        for chater in msgList:
            file.write("%s\n" % chater)
    
                
            
            
    
