import streamlit as st
import os
from rag_engine import get_rag_chain

st.set_page_config(page_title="PrivaShield Local RAG", page_icon="🛡️", layout="wide")

st.title("🛡️ PrivaShield: Local Compliance RAG")
st.subheader("Privacy-First Document Analysis powered by Gemma & Ollama")

# Sidebar for status and data preparation
with st.sidebar:
    st.header("Pipeline Management")
    if st.button("🔄 Initialize/Refresh Vector DB"):
        with st.spinner("Processing local PDFs..."):
            # Programmatically running ingest script
            os.system("python src/ingest.py")
            st.success("Vector DB Built successfully!")

# Check if DB exists
if not os.path.exists("chroma_db"):
    st.warning("⚠️ Vector Database not found. Please add PDFs to the 'data/' folder and click 'Initialize/Refresh Vector DB' in the sidebar.")
else:
    # Initialize RAG chain
    @st.cache_resource
    def load_chain():
        return get_rag_chain()
    
    try:
        rag_chain = load_chain()
        
        # User input query
        user_query = st.text_input("Ask a compliance, risk, or general question about your documents:")
        
        if user_query:
            with st.spinner("Analyzing locally..."):
                response = rag_chain.invoke({"input": user_query})
                
                # Display Answer
                st.markdown("### 🤖 Analysis & Answer")
                st.write(response["answer"])
                
                # Display Sources
                st.markdown("---")
                with st.expander("📚 View Retreived Source Context (Transparency)"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**Source Chunk {i+1} (Page {doc.metadata.get('page', 'N/A')}):**")
                        st.caption(doc.page_content)
    except Exception as e:
        st.error(f"Failed to connect to Ollama. Ensure `ollama serve` is running and you have pulled gemma. Error: {e}")
