# app.py
import streamlit as st
import sys
sys.path.append('.')

from src.pipeline import TeleRAGPipeline
from src.anomaly  import AnomalyDetector

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = 'TeleRAG',
    page_icon  = '📡',
    layout     = 'wide',
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .confidence-high   { color: #22c55e; font-weight: bold; }
    .confidence-medium { color: #f59e0b; font-weight: bold; }
    .confidence-low    { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title('📡 TeleRAG')
st.subheader('RAG-Based Future-Ready Telecom RAN Assistant')
st.markdown(
    'Domain-specific AI assistant for 3GPP specifications, O-RAN architecture, '
    'root cause analysis, and RAN anomaly detection.'
)
st.divider()


# ── Load pipeline (cached — runs only once per session) ───────
@st.cache_resource
def load_pipeline():
    with st.spinner('Loading TeleRAG pipeline (first load ~60 seconds)...'):
        pipeline = TeleRAGPipeline()
        detector = AnomalyDetector(pipeline)
    return pipeline, detector

pipeline, detector = load_pipeline()
st.success('✅ TeleRAG is ready')


# ── Session state ─────────────────────────────────────────────
if 'messages' not in st.session_state:
    st.session_state.messages = []


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header('⚙️ Settings')

    mode = st.radio(
        'Query Mode',
        options = [
            '💬 3GPP / O-RAN QnA',
            '🔍 Root Cause Analysis',
            '⚠️ Anomaly Detection',
        ],
        index = 0,
    )

    # CHANGE: cache toggle so user can force fresh answers
    use_cache = st.checkbox(
        'Use query cache',
        value = True,
        help  = 'Cached queries return instantly. Uncheck to force a fresh answer.',
    )

    st.divider()

    # Knowledge base stats
    st.markdown('**📚 Knowledge Base**')
    col1, col2 = st.columns(2)
    col1.metric('Vectors',   '~50K+')
    col2.metric('Specs',     '15+')
    col1.metric('TeleQnA',   '10K')
    col2.metric('O-RAN WGs', '2')

    st.divider()

    # Model info
    st.markdown('**🤖 Models**')
    st.markdown(
        '- **LLM:** tele-llm (LLaMA-3-8B telecom)\n'
        '- **Embed:** BGE-Large-v1.5\n'
        '- **Rerank:** ms-marco-MiniLM-L-6-v2\n'
        '- **Search:** FAISS + BM25 hybrid\n'
    )

    st.divider()

    if st.button('🗑️ Clear chat history'):
        st.session_state.messages = []
        st.rerun()


# ── Quick examples per mode ───────────────────────────────────
examples = {
    '💬 3GPP / O-RAN QnA': [
        'What is the purpose of PDCCH in 5G NR?',
        'What is the role of the RIC in O-RAN?',
        'Explain HARQ retransmissions in NR',
        'What does E2SM-KPM define in O-RAN?',
    ],
    '🔍 Root Cause Analysis': [
        'UE experiencing repeated handover failures in 5G NR',
        'High BLER and low throughput on uplink',
        'Random access failure after cell reselection',
        'Ping-pong handover between two cells',
    ],
    '⚠️ Anomaly Detection': [
        'CQI dropped from 12 to 3, BLER 40%, RLF on cell 47',
        'Sudden handover failure surge on sector 3',
        'UE throughput dropped 80%, high interference detected',
        'Cell outage — all KPIs degraded simultaneously',
    ],
}

st.markdown('**⚡ Quick examples:**')
cols = st.columns(2)
for i, q in enumerate(examples[mode]):
    if cols[i % 2].button(q, key=f'q{i}'):
        st.session_state['prefill'] = q


# ── Render chat history ───────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

        if msg['role'] == 'assistant':

            # CHANGE: confidence badge
            if 'confidence' in msg and msg['confidence']:
                conf  = msg['confidence']
                emoji = {'HIGH': '🟢', 'MEDIUM': '🟡', 'LOW': '🔴'}.get(conf, '⚪')
                st.markdown(
                    f'<span class="confidence-{conf.lower()}">'
                    f'{emoji} Confidence: {conf}</span>',
                    unsafe_allow_html=True,
                )

            # Anomaly badge
            if msg.get('is_anomaly') is not None:
                if msg['is_anomaly']:
                    st.error(
                        f"🚨 **Anomaly detected** — "
                        f"Keywords: {', '.join(msg.get('flagged_keywords', []))}"
                    )
                else:
                    st.success('✅ No anomaly detected')

            # CHANGE: sources with score bars
            if msg.get('sources'):
                with st.expander('📄 Sources used'):
                    for s in msg['sources']:
                        doc_type = s.get('type', 'N/A')
                        score    = s.get('score', 0)
                        bar_len  = int(score * 20)
                        bar      = '█' * bar_len + '░' * (20 - bar_len)
                        st.markdown(
                            f"**[{doc_type}]** `{s['source']}` "
                            f"— Page {s['page']}  \n"
                            f"`{bar}` {score:.3f}"
                        )

            # Timing
            if msg.get('timing'):
                st.caption(f"⏱️ {msg['timing']}")


# ── Chat input ────────────────────────────────────────────────
placeholders = {
    '💬 3GPP / O-RAN QnA':   'Ask a 3GPP or O-RAN question...',
    '🔍 Root Cause Analysis': 'Describe the RAN issue or fault...',
    '⚠️ Anomaly Detection':   'Paste RAN log or describe the metrics...',
}

user_input = st.chat_input(placeholders[mode])
if 'prefill' in st.session_state:
    user_input = st.session_state.pop('prefill')


# ── Process input ─────────────────────────────────────────────
if user_input:
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    with st.chat_message('assistant'):
        with st.spinner('Retrieving knowledge and generating answer...'):

            if mode == '💬 3GPP / O-RAN QnA':
                result          = pipeline.query(user_input, use_cache=use_cache)
                answer          = result['answer']
                sources         = result['sources']
                confidence      = result['confidence']
                timing          = (
                    f"Retrieval: {result['retrieval_time_s']}s  |  "
                    f"Generation: {result['generation_time_s']}s  |  "
                    f"Total: {result['total_time_s']}s"
                )
                is_anomaly       = None
                flagged_keywords = []

            elif mode == '🔍 Root Cause Analysis':
                result           = pipeline.rca_query(user_input)
                answer           = result['answer']
                sources          = result['sources']
                confidence       = result['confidence']
                timing           = 'Multi-step RCA (2 retrieval + 2 generation passes)'
                is_anomaly       = None
                flagged_keywords = []

            else:  # Anomaly Detection
                result           = detector.analyze(user_input)
                answer           = result['rag_analysis']
                sources          = result['sources']
                confidence       = 'HIGH' if result['is_anomaly'] else 'MEDIUM'
                timing           = 'Anomaly detection (keyword screening + RAG)'
                is_anomaly       = result['is_anomaly']
                flagged_keywords = result['flagged_keywords']

        # Render answer
        st.markdown(answer)

        # CHANGE: confidence badge
        emoji = {'HIGH': '🟢', 'MEDIUM': '🟡', 'LOW': '🔴'}.get(confidence, '⚪')
        st.markdown(
            f'<span class="confidence-{confidence.lower()}">'
            f'{emoji} Confidence: {confidence}</span>',
            unsafe_allow_html=True,
        )

        # Anomaly badge
        if is_anomaly is not None:
            if is_anomaly:
                st.error(
                    f"🚨 **Anomaly detected** — "
                    f"Keywords: {', '.join(flagged_keywords)}"
                )
            else:
                st.success('✅ No anomaly detected')

        # CHANGE: sources with visual score bars
        if sources:
            with st.expander('📄 Sources used'):
                for s in sources:
                    doc_type = s.get('type', 'N/A')
                    score    = s.get('score', 0)
                    bar_len  = int(score * 20)
                    bar      = '█' * bar_len + '░' * (20 - bar_len)
                    st.markdown(
                        f"**[{doc_type}]** `{s['source']}` "
                        f"— Page {s['page']}  \n"
                        f"`{bar}` {score:.3f}"
                    )

        st.caption(f"⏱️ {timing}")

    # Save to history
    st.session_state.messages.append({
        'role':             'assistant',
        'content':          answer,
        'sources':          sources,
        'confidence':       confidence,
        'timing':           timing,
        'is_anomaly':       is_anomaly,
        'flagged_keywords': flagged_keywords,
    })