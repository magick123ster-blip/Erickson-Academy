import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os
import sys

# 1. Setup Environment (Use relative paths)
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, 'erickson_vector_db')

st.set_page_config(page_title="Erickson Portable Training Center", page_icon="🎓", layout="wide")

# 2. Initialize Database (Portable)
@st.cache_resource
def get_collection():
    if not os.path.exists(db_path):
        st.error(f"데이터베이스를 찾을 수 없습니다: {db_path}")
        return None
    client = chromadb.PersistentClient(path=db_path)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return client.get_collection(name="erickson_strategies", embedding_function=emb_fn)

# 3. Sidebar State & Controls
st.sidebar.title("🎓 Erickson Academy")

# --- [NEW] Persona Dashboard ---
st.sidebar.markdown("---")
with st.sidebar.expander("👤 훈련 페르소나 및 상황 설정", expanded=False):
    st.markdown("### 심리-인류학적 프로필")
    p_profile = st.text_input("내담자 프로필", placeholder="예: 30세 여성, 음악가, 우울감")
    p_resistance = st.slider("저항 지수 (1-10)", 1, 10, 5)
    p_context = st.text_input("훈련 장소 및 상황", placeholder="예: 한낮의 카페, 대화형 코칭")
    p_conflict = st.text_area("갈등 상황 및 목적", placeholder="예: 무대 공포증 극복을 위한 자원 탐색")
    p_relationship = st.text_input("인간관계 체계", placeholder="예: 완벽주의적인 어머니와의 관계")
    use_persona = st.checkbox("페르소나 설정 반영", value=True)

persona_context = ""
if use_persona and (p_profile or p_context or p_conflict):
    persona_context = f"""
[🔒 IDENTITY LOCK: 당신의 정체성]
- 당신은 절대로 사용자를 가르치려 들거나 에릭슨인 척하지 마십시오.
- 당신은 지금부터 사용자가 에릭슨 기법으로 변화시켜야 할 **'상대방(내담자)'** 그 자체입니다.
- 사용자가 당신에게 말을 걸면, 아래의 프로필에 빙의하여 '현실적이고 까다로운' 반응을 보이십시오.

[👤 타겟 페르소나 프로필]
- 프로필: {p_profile}
- 저항 지수: {p_resistance}/10 (높을수록 기법에 쉽게 넘어가지 마십시오)
- 장소/상황: {p_context}
- 갈등 및 목적: {p_conflict}
- 인간관계 체계: {p_relationship}
"""
else:
    persona_context = "\n[🔒 IDENTITY LOCK] 당신은 사용자의 대화 상대(내담자)입니다. 사용자를 에릭슨 마스터로 대우하며, 당신은 변화가 필요한 평범한 혹은 저항적인 사람으로 대화에 임하십시오."

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 훈련 테마 선택")
training_themes = ["종합 마스터 통합 훈련(All-in-One)", "라포 및 동조", "저항 활용", "패턴 중단", "은유적 암시", "역설적 개입", "이중 구속", "혼란 기법", "미래 투사"]
selected_theme = st.sidebar.selectbox("연습할 전략", training_themes)

if st.sidebar.button("🚀 새 훈련 시작"):
    st.session_state.messages = []
    st.session_state.training_step = 0
    st.session_state.pending_trigger = f"나는 지금 '{selected_theme}' 전략을 훈련하고 싶어. 시나리오를 주고 1단계 미션을 줘."

st.sidebar.markdown("---")
api_key = st.sidebar.text_input("API Key 입력", type="password")
model_name = st.sidebar.selectbox("모델 선택", ["gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemma-4-31b-it"])

# 4. Master Statistical DNA
MASTER_DNA = """
[📊 에릭슨 통합 지능 데이터: 18,700+ 자산]
- 데이터: 공식, 콤보, 알고리즘, 아키타입이 통합된 1.87만 개 이상의 통계적 DNA.
- 훈련 모드: 실전 시퀀스 및 데이터 기반 전략 피드백 시스템.

[🧠 초정밀 추론 및 코칭 가이드]
- 추론 단계: 피드백 제공 전 최소 5단계 이상의 심층 전략 분석을 거칠 것.
- 답변 분량: 요약하지 마십시오. 사용자의 발화에서 발견된 패턴을 매우 상세하게 분석하고, 개선된 에릭슨식 답변을 풍성하게 제공하십시오.
- 데이터 활용: 검색된 'topic_id', 'cluster_id', 'srl_sample'을 근거로 사용자의 전략적 오류를 지적하고 교정하십시오.
"""

# 5. Trainer Logic
TRAINER_PROMPT = f"""
{MASTER_DNA}
{persona_context}

[📋 훈련 운영 규칙]
1. 대화의 시작: 선택된 테마({selected_theme})에 맞는 상황극 시나리오를 제시하고, 사용자가 첫마디를 떼도록 유도하십시오.
2. 대화 중(Persona Mode): 사용자의 발화에 대해 설정된 페르소나로서 '리얼하게' 반응하십시오. 저항 지수가 높다면 쉽게 설득되지 말고 에릭슨식 우회 기법을 쓰도록 유도하십시오.
3. 답변 구조: 모든 답변은 반드시 아래 형식을 따르십시오.
   ---
   [내담자 반응]
   (설정된 페르소나로서의 대화 내용)

   [🎓 마스터 피드백]
   (사용자의 방금 발화에 대한 에릭슨 전략적 분석, 잘한 점, 개선된 에릭슨식 대안 문장 제시)
   ---
4. 통합 훈련 모드: '종합 마스터 통합 훈련'인 경우, 모든 기법(은유, 혼란, 역설 등)을 복합적으로 요구하는 난해한 상황을 연출하십시오.
"""

# 6. Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "training_step" not in st.session_state:
    st.session_state.training_step = 0

# 7. UI
st.title("🎓 Erickson Professional Trainer (Portable)")
st.markdown(f"현재 모델: **{model_name}**")
st.markdown("---")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("에릭슨 스타일 문장 입력...")
if "pending_trigger" in st.session_state:
    prompt = st.session_state.pending_trigger
    del st.session_state.pending_trigger

if prompt:
    if "나는 지금" not in prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

    collection = get_collection()
    context = ""
    if collection:
        results = collection.query(query_texts=[prompt], n_results=3)
        for doc in results['documents'][0]: context += f"\n[DNA 사례]\n{doc}\n"
    
    if api_key:
        client = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=api_key)
        with st.chat_message("assistant"):
            full_response = ""
            # Build message history for continuity
            chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-10:]] # 최근 10턴 유지
            
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": TRAINER_PROMPT + f"\n[데이터]\n{context}"},
                    *chat_history,
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            placeholder = st.empty()
            for chunk in resp:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
            st.session_state.training_step += 1
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            if "나는 지금" in prompt:
                st.rerun()
