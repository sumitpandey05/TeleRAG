import streamlit as st
import sys
sys.path.append('.')
from src.pipeline import TeleRAGPipeline    

st.set_page_config(page_title='TeleRAG', page_icon='📡', layout='wide')
st.title('📡 TeleRAG')

st.subheader("RAG based Telecom RAN Assistant")
st.markdown('Ask any question about 3GPP specification or 5G NR.')
st.divider()

@st.cache_resource
def load_pipeline():
    with st.spinner('Loading TeleRAG (first load ~30 seconds)'):
        return TeleRAGPipeline()
    
pipeline = load_pipeline()
st.success('TeleRAG is ready')

if 'message' not in st.session_state:
    st.session_state.message = []

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        if 'sources' in msg:
            with st.expander('Sources used'):
                for s in msg['sources']:
                    st.write(f"{s['source']}, Page {s['page']} (score: {s['score']})")

examples = ['What is handover in 5G NR?', 'Explain the role of gNB',
            'What are HARQ retransmissions?', 'How does bearer setup work?']
st.markdown('**Quick questions:**')
cols = st.columns(2)
for i, q in enumerate(examples):
    if cols[i % 2].button(q, key=f'q{i}'):
        st.session_state['prefill'] = q
 
user_input = st.chat_input('Ask a telecom question...')
if 'prefill' in st.session_state:
    user_input = st.session_state.pop('prefill')
 
if user_input:
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)
    with st.chat_message('assistant'):
        with st.spinner('Searching and generating...'):
            result = pipeline.query(user_input)
        st.markdown(result['answer'])
        with st.expander('Sources used'):
            for s in result['sources']:
                st.write(f"{s['source']}, Page {s['page']} (score: {s['score']})")
    st.session_state.messages.append({
        'role': 'assistant', 'content': result['answer'], 'sources': result['sources']
    })
