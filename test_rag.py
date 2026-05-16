#Import the FAISS vector database class // In langchain, the FAISS vector database class is a wrapper that provides a high-performance interface for storing and searching high-dimensional embeddings using metas Facebook AI similarity search
from langchain_community.vectorstores import FAISS

# Import HuggingFace embedding model wrapper
# Embeddings convert text into numerical vectors
from langchain_huggingface import HuggingFaceEmbeddings

#Import OLlama LLM Wrapper
# This lets Python talk to your local Ollama model
from langchain_ollama import OllamaLLM as Ollama

# Import RetrievalQA chain
# This combines retrieval + LLM answering
from langchain.chains import RetrievalQA



print('Step 1: Loading Embedded model... ')

# Create embedding model object
# all-MiniLM-L6-v2 is a small fast embedding model // it is a lightweight sentence transformer that converts text into 384-dimensional dense vecotrs 
# It converts sentences into vectors
embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2') #name of the model we extract form hugging face
print('Done')

print('Step 2: Building vector store from test texts...')

#Create sample telecom Knowledge
texts = [
    '5G NR is the global standard for new generation mobile networks.',
    'A base station in 5G is called a gNB (gNodeB).',
    'Handover transfers an active session from one cell to another.',
    'HARQ stands for hybrid Automatic Repeat Request.',
]

#convert all the texts into embeddings 
#store them inside FAISS vector database 
vectorstore = FAISS.from_texts(texts, embeddings)
print('Done')

print('Step 3: Connecting to Ollama...')
#Create an LLM object connected to Ollama
llm = Ollama(model='llama3.1:8b', temperature=0) #name of te model, temperature means deterministic answers, lower temp means stable answers
print('Done')

print("Step 4: Running test question...")

#create Retrievla QA pipeline 
qa = RetrievalQA.from_chain_type (llm=llm, chain_type = 'stuff', retriever=vectorstore.as_retriever(search_kwargs={'k':2})) #llm used to generate final answer, #stuff type means the retrived answer is stuffed into the prompt, search_kwargs controlls retriever behaviour, k = 2 means top two relevant chunks

result = qa.invoke({'query': 'What is  handover in mobile networks?'})

print('\nAnswer: ', result['result'])
print('Setup is working correctly!')

print('\nSetup is working correctly')