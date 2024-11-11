import streamlit as st

from llm import get_ai_response


st.set_page_config(page_title="소득세 챗봇", page_icon="💰")

st.title("소득세 챗봇")
st.caption("소득세 관련 질문을 물어보세요.")


if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_question := st.chat_input("질문을 입력하세요."):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state['message_list'].append({"role": "user", "content": user_question})
    
    ai_response = get_ai_response(user_question)
    with st.chat_message("ai"):
        ai_message = st.write_stream(ai_response)
        st.session_state['message_list'].append({"role": "ai", "content": ai_message})


