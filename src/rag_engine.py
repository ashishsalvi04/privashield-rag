from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

DB_PATH = "chroma_db/"

def get_rag_chain():
    # 1. Load the existing vector database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    # 2. Create a retriever (Fetches top 3 relevant chunks)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 3. Initialize local Gemma via Ollama
    # Note: Ensure you have run `ollama run gemma` in your terminal first
    llm = Ollama(model="gemma")

    # 4. Define a professional, strict prompt system
    system_prompt = (
        "You are an expert AI Compliance and Legal Risk Assistant. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you do not know the answer based on the context, say clearly that "
        "the information is not available in the provided documents. Do not make things up.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 5. Build the RAG Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain