import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os
import sys
from sentence_transformers import CrossEncoder
from pydantic import BaseModel, Field
from typing import List
import langextract
from langextract.providers.gemini import GeminiLanguageModel
from langextract.providers.openai import OpenAILanguageModel

# 1. Setup Environment
current_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Erickson Portable Academy",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

    :root {
        --primary: #6366f1;
        --primary-hover: #4f46e5;
        --bg-dark: #0f172a;
        --card-bg: rgba(30, 41, 59, 0.7);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b, #0f172a);
    }

    /* Glassmorphism sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid var(--glass-border);
    }

    /* Custom Chat Input */
    .stChatInputContainer {
        padding-bottom: 20px;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
        border: none;
        color: white;
    }

    /* Cards/Expanders */
    .streamlit-expanderHeader {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 10px !important;
    }

    /* Status Info */
    .stAlert {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid var(--glass-border) !important;
        color: #e2e8f0 !important;
    }

    /* Headings */
    h1, h2, h3 {
        background: linear-gradient(to right, #fff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }

    /* Tables */
    .stTable {
        background-color: var(--card-bg);
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# 3. Initialize Database (Portable)
@st.cache_resource
def get_rag_tools():
    # Attempt to find the correct database path
    possible_paths = [
        os.path.join(current_dir, 'erickson_vector_db', 'erickson_vector_db'),
        os.path.join(current_dir, 'erickson_vector_db'),
        os.path.join(current_dir, 'erickson_vector_db_v2')
    ]
    
    db_path = None
    for p in possible_paths:
        if os.path.exists(os.path.join(p, 'chroma.sqlite3')):
            db_path = p
            break
            
    if not db_path:
        st.error("Vector Database not found. Please check the 'erickson_vector_db' folder.")
        return None, None
        
    try:
        client = chromadb.PersistentClient(path=db_path)
        # 1. Faster high-resolution embedding model
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        
        # Try to find the collection name
        collections = client.list_collections()
        if not collections:
             st.error("No collections found in database.")
             return None, None
             
        # Prefer 'erickson_strategies_v2' but fall back to the first one available
        collection_name = "erickson_strategies_v2"
        existing_names = [c.name for c in collections]
        if collection_name not in existing_names:
            collection_name = existing_names[0]
            
        collection = client.get_collection(name=collection_name, embedding_function=emb_fn)
        
        # 2. Local Reranker (Cross-Encoder)
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        return collection, reranker
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return None, None

# 4. LangExtract Schema for Training Analysis
class TrainingStrategyAnalysis(BaseModel):
    technique_name: str = Field(description="The Ericksonian technique identified in the interaction.")
    pedagogical_note: str = Field(description="Why this technique is important for the student to learn.")
    grounded_quote: str = Field(description="Direct reference from Erickson's master data.")

class TrainingFullAnalysis(BaseModel):
    key_takeaways: List[TrainingStrategyAnalysis]
    student_progress: str = Field(description="Brief assessment of the student's mastery in this turn.")

# 5. LangExtract Examples
def get_training_examples():
    from langextract.data import ExampleData, Extraction
    return [
        ExampleData(
            text="🎓 [마스터 피드백]: 훈련생님, 방금 사용하신 '미러링' 기법은 내담자와의 라포를 형성하는 데 매우 효과적이었습니다.",
            extractions=[
                Extraction(
                    extraction_class="Mirroring",
                    extraction_text="방금 사용하신 '미러링' 기법",
                    attributes={"note": "내담자의 비언어적 행동을 동조하여 신뢰를 구축함"}
                )
            ]
        )
    ]

# 6. Sidebar UI
st.sidebar.markdown("<h1 style='text-align: center; color: white;'>🎓 Erickson Academy</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")

with st.sidebar.expander("👤 훈련 페르소나 설정", expanded=False):
    p_profile = st.text_input("내담자 프로필", value="30세 여성, 음악가, 우울감")
    p_resistance = st.slider("저항 지수 (1-10)", 1, 10, 5)
    p_context = st.text_input("장소 및 상황", value="한낮의 카페, 대화형 코칭")
    p_conflict = st.text_area("갈등 및 목적", value="무대 공포증 극복을 위한 자원 탐색")
    p_relationship = st.text_input("인간관계 체계", value="완벽주의적인 어머니와의 관계")
    use_persona = st.checkbox("페르소나 설정 반영", value=True)

persona_context = ""
if use_persona:
    persona_context = f"""
[🔒 IDENTITY LOCK: 당신의 정체성]
- 당신은 절대로 사용자를 가르치려 들거나 에릭슨인 척하지 마십시오.
- 당신은 지금부터 사용자가 에릭슨 기법으로 변화시켜야 할 **'상대방(내담자)'** 그 자체입니다.
- 사용자가 당신에게 말을 걸면, 아래의 프로필에 빙의하여 '현실적이고 까다로운' 반응을 보이십시오.

[👤 타겟 페르소나 프로필]
- 프로필: {p_profile}
- 저항 지수: {p_resistance}/10
- 장소/상황: {p_context}
- 갈등 및 목적: {p_conflict}
- 인간관계 체계: {p_relationship}
"""
else:
    persona_context = "\n[🔒 IDENTITY LOCK] 당신은 사용자의 대화 상대(내담자)입니다. 사용자를 에릭슨 마스터로 대우하며, 평범한 대화를 이어가십시오."

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 훈련 테마")
training_themes = ["종합 마스터 통합 훈련(All-in-One)", "라포 및 동조", "저항 활용", "패턴 중단", "은유적 암시", "역설적 개입", "이중 구속", "혼란 기법", "미래 투사"]
selected_theme = st.sidebar.selectbox("연습할 전략", training_themes)

if st.sidebar.button("🚀 새 훈련 시작"):
    st.session_state.messages = []
    st.session_state.training_step = 0
    st.session_state.pending_trigger = f"나는 지금 '{selected_theme}' 전략을 훈련하고 싶어. 시나리오를 주고 1단계 미션을 줘."

st.sidebar.markdown("---")
api_key = st.sidebar.text_input("Google API Key", type="password", placeholder="AI Studio에서 발급받은 키")
model_name = st.sidebar.selectbox("모델 선택", ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-exp"])

# 7. Master System Prompt
MASTER_DNA = """
[📊 에릭슨 통합 지능 데이터: 18,700+ 자산]
- 당신은 밀턴 에릭슨의 전략적 소통 DNA를 완벽하게 이해하고 있습니다.

<reasoning_protocol>
1. [Data Anchoring]: 검색된 사례(DNA 사례)를 바탕으로 응답하십시오.
2. [Diagnostic]: 사용자의 발화에서 에릭슨 기법의 오류나 부족한 점을 분석하십시오.
</reasoning_protocol>

<output_instruction>
- 절대로 요약하지 마십시오.
- [🎓 마스터 피드백] 섹션에서는 사용자의 패턴을 아주 상세하게 분석하십시오.
- 구체적이고 복합적인 에릭슨 스타일의 대안 문장을 제시하십시오.
</output_instruction>
"""

TRAINER_PROMPT = f"""
{MASTER_DNA}
{persona_context}

[📋 훈련 운영 규칙]
1. 시나리오 제시: 선택된 테마({selected_theme})에 맞춰 상황극을 시작하십시오.
2. 페르소나 모드: 사용자의 말에 내담자로서 반응하되, 저항 지수에 맞춰 쉽게 넘어가지 마십시오.
3. 답변 구조 필수:
   ---
   [내담자 반응]
   (내담자로서의 대사)

   [🎓 마스터 피드백]
   (상세 분석 및 대안 제시)
   ---
"""

# 8. Main UI
st.markdown("<h1 style='text-align: center;'>🎓 Erickson Portable Academy</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>Mastering the Art of Hypnotic Communication</p>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Logic
prompt = st.chat_input("에릭슨 스타일로 대답해 보세요...")
if "pending_trigger" in st.session_state:
    prompt = st.session_state.pending_trigger
    del st.session_state.pending_trigger

if prompt:
    if "나는 지금" not in prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

    collection, reranker = get_rag_tools()
    context = ""
    if collection and reranker:
        results = collection.query(query_texts=[prompt], n_results=15)
        if results and results['documents'] and len(results['documents'][0]) > 0:
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            pairs = [[prompt, doc] for doc in docs]
            scores = reranker.predict(pairs)
            scored_docs = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)
            top_results = scored_docs[:5]
            for score, doc, meta in top_results:
                context += f"\n[DNA 사례] (유사도: {score:.2f})\n{doc}\n"

    if api_key:
        try:
            client = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=api_key)
            with st.chat_message("assistant"):
                full_response = ""
                chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-10:]]
                
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": f"[참조 전략 데이터]\n{context}\n\n{TRAINER_PROMPT}"},
                        *chat_history,
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    stream=True
                )
                
                placeholder = st.empty()
                for chunk in resp:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

                # LangExtract Analysis
                try:
                    with st.expander("📈 실전 전략 분석 보고서"):
                        le_model = GeminiLanguageModel(model_id=model_name, api_key=api_key)
                        from langextract.data import FormatType
                        analysis_result = langextract.extract(
                            text_or_documents=full_response,
                            prompt_description="Analyze the Ericksonian techniques used in the 마스터 피드백 section. Provide names and notes in Korean.",
                            examples=get_training_examples(),
                            model=le_model,
                            format_type=FormatType.YAML,
                            fence_output=True
                        )
                        if analysis_result and analysis_result.extractions:
                            extracted_data = []
                            for ex in analysis_result.extractions:
                                extracted_data.append({
                                    "기법": ex.extraction_class,
                                    "교육적 조언": ex.attributes.get("note", "분석 중...") if ex.attributes else "분석 중...",
                                    "핵심 내용": ex.extraction_text
                                })
                            st.table(extracted_data)
                except:
                    pass
        except Exception as e:
            st.error(f"API Error: {str(e)}")
    else:
        st.warning("왼쪽 사이드바에 Google API Key를 입력해 주세요.")

    if "나는 지금" in prompt:
        st.rerun()
