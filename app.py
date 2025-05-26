import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
import os
# 

load_dotenv()

# Streamlit UI
st.title("🔎 LangChain - Chat with Search Agent")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search Arxiv, Wikipedia, and the web. How can I help you?"}
    ]

# Display message history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Set up tools
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)

    tools = [
        DuckDuckGoSearchRun(name="Search"),
        ArxivQueryRun(api_wrapper=arxiv_wrapper),
        WikipediaQueryRun(api_wrapper=wiki_wrapper),
    ]

    # Load LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

    # Initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            response = agent.run(prompt, callbacks=[st_cb])
        except Exception as e:
            response = f"❌ An error occurred: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
