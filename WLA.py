import streamlit as st
import os
from langchain_community.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# --- Streamlit 페이지 설정 및 초기화 ---
st.set_page_config(
    page_title="Agilent AI 어시스턴트",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- OpenAI API 키 관리 (st.secrets 사용) ---
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = st.secrets.get("openai_api_key")

# --- 사용자 정의 CSS 스타일 적용 ---
st.markdown(
    """
    <style>
    .main-header { font-size: 3em; font-weight: bold; color: #007bff; text-align: center; margin-bottom: 20px; padding-top: 10px; }
    .stButton>button { background-color: #007bff; color: white; border-radius: 5px; padding: 10px 20px; font-size: 1.2em; border: none; transition: background-color 0.2s; }
    .stButton>button:hover { background-color: #0056b3; color: white; }
    .stTextInput>div>div>input { border-radius: 5px; border: 1px solid #ced4da; padding: 10px; font-size: 1em; }
    .stInfo { background-color: #e2f0fb; border-left: 5px solid #007bff; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-size: 1.1em; }
    .stSuccess { background-color: #d4edda; border-left: 5px solid #28a745; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-size: 1.1em; }
    .stWarning { background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-size: 1.1em; }
    .stError { background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-size: 1.1em; }
    </style>
    """, unsafe_allow_html=True
)

# --- 메인 화면 UI 구성 ---
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://i.imgur.com/rN5V9aW.png" alt="Agilent Logo" style="height: 50px;">
        <h1 class='main-header' style='display: inline-block; margin-left: 10px; vertical-align: middle;'>Agilent 문서 Q&A</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# --- 사이드바 UI 구성 ---
with st.sidebar:
    st.image("agilent_logo2.png", width=130)
    st.title("Agilent AI 어시스턴트")
    st.markdown(
        """
        이 앱은 Agilent 기술 문서를 기반으로 질문에 답변합니다.
        궁금한 점이 있으시면 언제든지 질문해주세요!
        """
    )
    st.markdown("---")
    st.markdown("Developed by Agilent AI Team")

# --- 데이터 로드 및 처리 ---
@st.cache_resource
def load_and_process_data():
    if not os.path.exists("Address.txt"):
        return "file_error"
    try:
        with open("Address.txt", 'r', encoding='utf-8') as f:
            text_data = f.read()
    except Exception as e:
        return f"read_error: {e}"
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(text_data)
    with st.spinner("임베딩 모델을 로드 중입니다..."):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            return f"embedding_error: {e}"
    docs = FAISS.from_texts(texts, embeddings)
    docsearch = docs.as_retriever()
    return docsearch

doc_load_result = load_and_process_data()

if isinstance(doc_load_result, str):
    if "file_error" in doc_load_result:
        st.error("오류: 'Address.txt' 파일을 찾을 수 없습니다. 파일을 앱이 실행되는 폴더에 넣어주세요.")
    elif "read_error" in doc_load_result:
        st.error(f"오류: 'Address.txt' 파일을 읽는 중 문제가 발생했습니다: {doc_load_result.split(': ')[1]}")
    elif "embedding_error" in doc_load_result:
        st.error(f"오류: 임베딩 모델을 로드하는 데 실패했습니다. 인터넷 연결을 확인하거나 모델 이름을 확인해주세요. ({doc_load_result.split(': ')[1]})")
    st.stop()
else:
    docsearch = doc_load_result
    st.success("데이터베이스 준비 완료! 질문을 입력해주세요.")

# --- LLM 및 RAG 체인 실행 ---
user_query = st.text_input("질문을 입력하세요:", placeholder="예: '---정보 알려줘.'", key="user_query_input")

if user_query and st.session_state.openai_api_key:
    try:
        llm = OpenAI(temperature=0.7, openai_api_key=st.session_state.openai_api_key)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch, return_source_documents=False)
    except Exception as e:
        st.error(f"LLM 초기화 또는 RAG 체인 설정 중 오류가 발생했습니다. API 키를 확인해주세요. ({e})")
        llm = None
    if llm:
        with st.spinner("답변을 생성하는 중입니다. 잠시만 기다려 주세요..."):
            try:
                with st.spinner("AI 비서가 기술 자료를 검색하고 있습니다..."):
                    result = qa.run(user_query)
                st.success("답변 생성 완료! 🚀")
                st.subheader("생성된 답변")
                st.info(result)
            except Exception as e:
                st.error(f"질문 처리 중 오류가 발생했습니다: {e}")
                st.error("OpenAI API 키가 올바른지, 또는 사용량 제한에 도달했는지 확인해주세요.")
                
elif user_query and not st.session_state.openai_api_key:
    st.warning("OpenAI API 키가 필요합니다. Secrets에 키를 설정해주세요.")
elif not user_query and not isinstance(doc_load_result, str):
    st.info("안녕하세요! 애질런트 문서에 대해 궁금한 점을 물어보세요.")