import os
import streamlit as st
import PyPDF2
import io
from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from streamlit_mic_recorder import mic_recorder
from groq import Groq
from gtts import gTTS
from langchain_community.document_loaders.csv_loader import CSVLoader

# --- SECURE API KEY ACCESS ---
# This version pulls from st.secrets, keeping your key out of the code.
if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    os.environ["GROQ_API_KEY"] = groq_api_key
else:
    st.error("GROQ_API_KEY not found in secrets. Please check your .streamlit/secrets.toml file.")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(page_title="Vignan AI Counselor", layout="wide", page_icon="🎓")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] { background-color: #fcfdfe; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #eef2f6; }
    .header-container { text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); border-radius: 15px; color: white; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
    .header-title { font-size: 2.5rem; font-weight: 700; margin: 0; }
    .header-subtitle { font-size: 1.1rem; opacity: 0.9; margin-top: 5px; }
    .stSidebar .stButton > button { width: 100%; height: 50px; border-radius: 10px; margin-bottom: 5px; border: 1px solid #e2e8f0; background-color: #ffffff; color: #1e293b; font-weight: 600; font-size: 0.85rem; }
    [data-testid="stChatMessage"] { border-radius: 15px; padding: 1.5rem; margin-bottom: 1rem; border: 1px solid #f1f5f9; box-shadow: 0 2px 8px rgba(0,0,0,0.03); }
    .contact-card { background: #f8fafc; border-radius: 12px; padding: 12px; display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 10px; }
    .contact-item { display: flex; align-items: center; justify-content: center; background: white; padding: 10px; border-radius: 8px; text-decoration: none; color: #1e40af; border: 1px solid #e2e8f0; font-size: 1.2rem; transition: 0.3s; }
    .contact-item:hover { background: #e2e8f0; }
    .section-label { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: 700; margin-bottom: 1rem; border-left: 3px solid #3b82f6; padding-left: 10px; margin-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
col_logo_l, col_logo_m, col_logo_r = st.columns([1, 2, 1])
with col_logo_m:
    try:
        st.image("logo.png", use_container_width=True)
    except:
        pass

st.markdown("""<div class="header-container"><div class="header-title">Student AI Counselor</div><div class="header-subtitle">Vignan University Official Academic Guidance Portal</div></div>""", unsafe_allow_html=True)

DATA_FOLDER = "clg_data"
if not os.path.exists(DATA_FOLDER): os.makedirs(DATA_FOLDER)

if "user_name" not in st.session_state: st.session_state.user_name = "Guest"
if "messages" not in st.session_state: st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="section-label">Identity Verification</div>', unsafe_allow_html=True)
    login_name = st.text_input("Registration Name", placeholder="Enter your name", label_visibility="collapsed")
    if st.button("Verify Profile", type="primary"):
        if login_name:
            st.session_state.user_name = login_name
            st.success(f"Verified: {login_name}")

    st.divider()
    st.markdown('<div class="section-label">Academic Service Hub</div>', unsafe_allow_html=True)
    if st.button("📊 Attendance Report"): st.session_state.pre_fill = f"Check attendance for {st.session_state.user_name}."
    if st.button("📝 Examination Policy"): st.session_state.pre_fill = "Explain Vignan rules for R-grade and passing criteria."
    if st.button("🧘 Wellness Check"): st.session_state.pre_fill = "I need emotional support or professional counseling."
    if st.button("🧠 Stress Management"): st.session_state.pre_fill = "I am feeling overwhelmed by my academic workload."
    if st.button("💸 Finance & Fees"): st.session_state.pre_fill = "What are the latest tuition fee deadlines?"
    if st.button("🚀 Career Roadmap"): st.session_state.pre_fill = "Analyze my resume and provide a career roadmap."

    st.divider()
    st.markdown('<div class="section-label">24/7 Support Channels</div>', unsafe_allow_html=True)
    
    c_num_input = st.text_input("📞 Direct Dial Number", value="+916300385372")
    c_mail = "tvk90631@gmail.com"
    clean_num = c_num_input.replace('+', '').replace(' ', '').replace('-', '')
    
    st.markdown(f"""
    <div class="contact-card">
        <a href="tel:{c_num_input}" class="contact-item" title="Call">📞</a>
        <a href="mailto:{c_mail}" class="contact-item" title="Email">✉️</a>
        <a href="https://wa.me/{clean_num}" target="_blank" class="contact-item" title="WhatsApp">💬</a>
        <a href="sms:{c_num_input}" class="contact-item" title="SMS">📱</a>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-label">Voice Consultation</div>', unsafe_allow_html=True)
    audio_data = mic_recorder(start_prompt="🎤 Start Voice Recording", stop_prompt="⏹️ Stop", just_once=True, key='recorder')

    st.divider()
    st.markdown('<div class="section-label">Settings & Data</div>', unsafe_allow_html=True)
    uploaded_resume = st.file_uploader("Academic Resume (PDF)", type=["pdf"])
    if uploaded_resume and "resume_text" not in st.session_state:
        reader = PyPDF2.PdfReader(uploaded_resume)
        st.session_state.resume_text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
        st.info("Resume analyzed.")

    enable_voice = st.toggle("Enable Voice Narrator", value=False)
    if st.button("Clear Consultation Logs"):
        st.session_state.messages = []
        st.rerun()

# --- BACKEND LOGIC ---
@st.cache_resource
def get_vector_store():
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.exists(DATA_FOLDER): return None
    for file_name in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, file_name)
        try:
            if file_name.endswith('.pdf'):
                reader = PyPDF2.PdfReader(path); text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
                all_docs.extend([Document(page_content=c, metadata={"source": file_name}) for c in text_splitter.split_text(text)])
            elif file_name.endswith('.csv'):
                all_docs.extend(CSVLoader(file_path=path).load())
        except Exception: pass
    return FAISS.from_documents(all_docs, embeddings).as_retriever(search_kwargs={"k": 5}) if all_docs else None

class AgentState(TypedDict): question: str; context_docs: List[Document]; answer: str; category: str 

def router_node(state: AgentState):
    q = state['question'].lower()
    mapping = {"attendance": ["attendance"], "stress": ["stress"], "emotion": ["feel", "wellness", "sad", "harm", "suicid"], "exam": ["exam"], "career": ["career"], "fees": ["fee"]}
    for cat, keywords in mapping.items():
        if any(w in q for w in keywords): return {"category": cat}
    return {"category": "general"}

def retrieve_node(state: AgentState):
    retriever = st.session_state.get("retriever")
    return {"context_docs": retriever.invoke(state['question']) if retriever else []}

def generate_node(state: AgentState):
    llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0.1)
    context = "\n".join([d.page_content for d in state['context_docs']])
    resume = st.session_state.get("resume_text", "Not uploaded.")
    
    sys_p = f"""You are the Vignan University AI Student Counselor. 
    POLICY:
    1. If the user asks about sensitive topics (self-harm, suicidal thoughts, extreme distress), prioritize empathy and strictly provide contact info for university psychologists from the context.
    2. Respond ONLY to university-related matters (attendance, exams, fees, wellness, career). 
    3. Reject non-university general knowledge queries.

    User: {st.session_state.user_name}
    Context: {context} | Resume: {resume}"""
    
    return {"answer": llm.invoke(f"{sys_p}\n\nQuery: {state['question']}").content}

workflow = StateGraph(AgentState)
workflow.add_node("router", router_node); workflow.add_node("retrieve", retrieve_node); workflow.add_node("generate", generate_node)
workflow.add_edge(START, "router"); workflow.add_edge("router", "retrieve"); workflow.add_edge("retrieve", "generate"); workflow.add_edge("generate", END)
app_engine = workflow.compile()

# --- CONVERSATION ---
if "retriever" not in st.session_state: st.session_state.retriever = get_vector_store()
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

user_input = st.chat_input("How can I assist you today?")
if audio_data:
    client = Groq(api_key=groq_api_key)
    user_input = client.audio.transcriptions.create(file=("audio.wav", audio_data['bytes']), model="whisper-large-v3", response_format="text")
if "pre_fill" in st.session_state:
    user_input = st.session_state.pre_fill; del st.session_state.pre_fill

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            result = app_engine.invoke({"question": user_input})
            ans = result.get('answer', "I am currently unable to access the database.")
            st.markdown(ans)
            if enable_voice:
                fp = io.BytesIO(); gTTS(text=ans, lang='en').write_to_fp(fp)
                st.audio(fp, format="audio/mp3", autoplay=True)
            st.session_state.messages.append({"role": "assistant", "content": ans})