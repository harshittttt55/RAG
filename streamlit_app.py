import streamlit as st
from Rag_Pipeline import RAGBOT

# ---------------------------
# Streamlit App Configuration
# ---------------------------
st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🤖 RAG Chatbot")

# ---------------------------
# Initialize Models & Vector Store
# ---------------------------
@st.cache_resource(show_spinner=True)
def initialize_rag():
    llm, embeddings = RAGBOT.initialize_models()
    vector_store, collection_name = RAGBOT.setup_vector_store(embeddings)
    rag_chain = RAGBOT.create_rag_chain(vector_store, llm)
    return rag_chain

rag_chain = initialize_rag()

# ---------------------------
# Initialize session state
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # List of tuples: (question, answer)

# ---------------------------
# Layout: Left = Query/Results, Right = History
# ---------------------------
left_col, right_col = st.columns([3, 1])

with left_col:
    question = st.text_input("Ask a question:", key="input_question")

    if st.button("Submit") and question.strip():
        # Run the RAG chain
        with st.spinner("Thinking..."):
            answer = ""
            for chunk in rag_chain.stream(question):
                answer += chunk

        # Display answer
        st.markdown("**Answer:**")
        st.write(answer)

        # Save to history
        st.session_state.history.append((question, answer))

with right_col:
    st.markdown("### 🕘 Search History")
    if st.session_state.history:
        for idx, (q, a) in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"**{idx}. Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")
    else:
        st.info("No history yet.")
