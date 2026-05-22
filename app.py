# app.py
import streamlit as st
import sys
sys.path.append('.')

from src.pipeline import TeleRAGPipeline
from src.anomaly import AnomalyDetector

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = 'TeleRAG',
    page_icon  = '📡',
    layout     = 'wide',
)

st.title('📡 TeleRAG')
st.subheader('RAG-Based Telecom RAN Assistant')
st.markdown('Ask questions about 3GPP specifications, O-RAN architecture, perform root cause analysis, or detect anomalies.')
st.divider()


# ── Load pipeline (cached — only runs once) ───────────────────
@st.cache_resource
def load_pipeline():
    with st.spinner('Loading TeleRAG pipeline (first load ~30 seconds)...'):
        pipeline = TeleRAGPipeline()
        detector = AnomalyDetector(pipeline)
    return pipeline, detector

pipeline, detector = load_pipeline()
st.success('✅ TeleRAG is ready')


# ── FIX 1: was 'message' — typo caused KeyError on every run ──
if 'messages' not in st.session_state:
    st.session_state.messages = []


# ── Sidebar — mode selector ───────────────────────────────────
with st.sidebar:
    st.header('⚙️ Settings')

    mode = st.radio(
        'Query Mode',
        options   = ['💬 QnA', '🔍 Root Cause Analysis', '⚠️ Anomaly Detection'],
        index     = 0,
        help      = (
            'QnA: standard question answering from 3GPP/O-RAN specs.\n'
            'RCA: multi-step reasoning for fault diagnosis.\n'
            'Anomaly: detect faults in RAN logs.'
        )
    )

    st.divider()
    st.markdown('**About**')
    st.markdown(
        '- 📚 18,931 indexed vectors\n'
        '- 📄 3GPP TS 38.211/213/214/300/321/331/401\n'
        '- 📡 O-RAN WG1/WG3 specifications\n'
        '- ❓ 10,000 TeleQnA Q&A pairs\n'
    )

    if st.button('🗑️ Clear chat history'):
        st.session_state.messages = []
        st.rerun()


# ── Quick example questions per mode ─────────────────────────
examples = {
    '💬 QnA': [
        'What is the purpose of PDCCH in 5G NR?',
        'Explain the role of gNB in NG-RAN',
        'What are HARQ retransmissions?',
        'What is the role of the RIC in O-RAN?',
    ],
    '🔍 Root Cause Analysis': [
        'UE experiencing repeated handover failures in 5G NR',
        'High packet loss observed on uplink in cell 12',
        'Random access failure after cell reselection',
        'Throughput degradation after software upgrade',
    ],
    '⚠️ Anomaly Detection': [
        'CQI dropped from 12 to 3, BLER spiked to 40%, RLF on cell 47',
        'Sudden increase in handover failures on sector 3',
        'UE throughput dropped 80% with high interference detected',
        'Cell outage reported, all KPIs degraded simultaneously',
    ],
}

st.markdown('**Quick examples:**')
cols = st.columns(2)
for i, q in enumerate(examples[mode]):
    if cols[i % 2].button(q, key=f'q{i}'):
        st.session_state['prefill'] = q


# ── Render chat history ───────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

        # Show sources for assistant messages
        if msg['role'] == 'assistant' and 'sources' in msg and msg['sources']:
            with st.expander('📄 Sources used'):
                for s in msg['sources']:
                    # FIX 3: show type field alongside source name
                    doc_type = s.get('type', 'N/A')
                    st.write(
                        f"**[{doc_type}]** {s['source']} "
                        f"— Page {s['page']} "
                        f"(score: {s['score']})"
                    )

        # Show timing for assistant messages
        if msg['role'] == 'assistant' and 'timing' in msg:
            st.caption(f"⏱️ {msg['timing']}")

        # Show anomaly flag if present
        if msg['role'] == 'assistant' and 'is_anomaly' in msg:
            if msg['is_anomaly']:
                st.error(f"🚨 Anomaly detected — keywords: {', '.join(msg['flagged_keywords'])}")
            else:
                st.success('✅ No anomaly detected')


# ── Chat input ────────────────────────────────────────────────
placeholder_map = {
    '💬 QnA':                 'Ask a 3GPP or O-RAN question...',
    '🔍 Root Cause Analysis': 'Describe the RAN issue or fault...',
    '⚠️ Anomaly Detection':   'Paste RAN log or describe the metrics...',
}

user_input = st.chat_input(placeholder_map[mode])

# Handle quick-example button prefill
if 'prefill' in st.session_state:
    user_input = st.session_state.pop('prefill')


# ── Process input ─────────────────────────────────────────────
if user_input:

    # Add user message to history
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    # Generate response based on mode
    with st.chat_message('assistant'):
        with st.spinner('Retrieving and generating...'):

            if mode == '💬 QnA':
                result      = pipeline.query(user_input)
                answer      = result['answer']
                sources     = result['sources']
                timing      = (
                    f"Retrieval: {result['retrieval_time_s']}s  |  "
                    f"Generation: {result['generation_time_s']}s  |  "
                    f"Total: {result['total_time_s']}s"
                )
                is_anomaly      = None
                flagged_keywords = []

            elif mode == '🔍 Root Cause Analysis':
                result      = pipeline.rca_query(user_input)
                answer      = result['answer']
                sources     = result['sources']
                timing      = 'Multi-step RCA (2 retrieval + 2 generation passes)'
                is_anomaly       = None
                flagged_keywords = []

            else:  # Anomaly Detection
                result           = detector.analyze(user_input)
                answer           = result['rag_analysis']
                sources          = result['sources']
                timing           = 'Anomaly detection (keyword + RAG analysis)'
                is_anomaly       = result['is_anomaly']
                flagged_keywords = result['flagged_keywords']

        # Render answer
        st.markdown(answer)

        # Show anomaly badge
        if is_anomaly is not None:
            if is_anomaly:
                st.error(f"🚨 Anomaly detected — keywords: {', '.join(flagged_keywords)}")
            else:
                st.success('✅ No anomaly detected')

        # Show sources
        if sources:
            with st.expander('📄 Sources used'):
                for s in sources:
                    doc_type = s.get('type', 'N/A')
                    st.write(
                        f"**[{doc_type}]** {s['source']} "
                        f"— Page {s['page']} "
                        f"(score: {s['score']})"
                    )

        # Show timing
        st.caption(f"⏱️ {timing}")

    # Save assistant message to history
    st.session_state.messages.append({
        'role':             'assistant',
        'content':          answer,
        'sources':          sources,
        'timing':           timing,
        'is_anomaly':       is_anomaly,
        'flagged_keywords': flagged_keywords,
    })