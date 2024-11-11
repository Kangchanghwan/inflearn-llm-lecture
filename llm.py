from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from answer_exam import answer_examples

load_dotenv()
store = {}


def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model_name="gpt-4o")
    return llm

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_vector_store(): 
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = "tax-markdown-index"

    return PineconeVectorStore.from_existing_index(
        index_name,
        embedding=embeddings,
    )    
def get_history_retriever():
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ]
    )
    return create_history_aware_retriever(
        get_llm(),
        get_vector_store().as_retriever(),
        contextualize_q_prompt
    )

def get_rag_chain():

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}")
        ]
    )

    few_shot_promt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples
    )

    qa_system_prompt = (
    "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변을 해주세요."
    "아래에 제공된 문서를 활용해서 답변해 주시고"
    "답변을 알 수 없다면 모른다고 답변해 주세요"
    "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고"
    "2-3 문장정도의 짧은 내용의 답변을 원합니다."
    "{context}")
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            few_shot_promt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(get_llm(), qa_prompt)
    rag_chain = create_retrieval_chain(get_history_retriever(), question_answer_chain)


    converation_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')

    return converation_rag_chain

def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    llm = get_llm()
    promt = ChatPromptTemplate.from_template(f"""
                                            사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
                                            만약 변경할 필요가 없다고 판단된디면, 사용자의 질문을 변경하지 않아도 됩니다.
                                            그런 경우에는 질문만 반환해주세요.
                                            사전: {dictionary}

                                            질문: {{input}}
                                            """)

    dictionary_chain = promt | llm | StrOutputParser()
    return dictionary_chain


def get_ai_response(ai_question):
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()

    tax_chain = {"input": dictionary_chain} | rag_chain

    ai_response = tax_chain.stream(
        input={"input": ai_question}, 
        config={'configurable': {'session_id': 'google'}}
    )

    return ai_response