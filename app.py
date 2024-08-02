from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from more_itertools import chunked

from langserve import RemoteRunnable

# --- Logo ---
st.set_page_config(
    page_title="Minhaj University Chatbot",
    page_icon="https://www.mul.edu.pk/images/minhaj-university-lahore.png",
    layout="wide",
)

st.sidebar.image("https://www.mul.edu.pk/images/logo-mul.png", width=200)

st.sidebar.title("Navigation")
st.sidebar.write("DISCIPLINE-INNOVATION-EXCELLENCE-CHARITY")
st.sidebar.markdown("[Visit Minhaj University Lahore](https://www.mul.edu.pk/en/)")


rag_chain = RemoteRunnable("http://69.61.24.171:8000/rag_chain/")


prompt = ChatPromptTemplate(
    messages=[
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

msgs = StreamlitChatMessageHistory(key="langchain_messages")

# --- Main Content ---
st.markdown("## 🔍 Chatbot For Minhaj University Lahore:")

if len(msgs.messages) == 0:
    msgs.add_ai_message("Hello! How can I assist you today?")


for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    st.chat_message("human").write(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            _chat_history = st.session_state.langchain_messages[1:40]
            _chat_history_tranform = list(
                chunked([msg.content for msg in _chat_history], n=2)
            )

            response = rag_chain.stream(
                {"question": prompt, "chat_history": _chat_history_tranform}
            )

            for res in response:
                full_response += res or ""
                message_placeholder.markdown(full_response + "|")
                message_placeholder.markdown(full_response)

            msgs.add_user_message(prompt)
            msgs.add_ai_message(full_response)

        except Exception as e:
            st.error(f"An error occured. {e}")
