# Healthcare Professional Assistant - Frontend using Streamlit + Ollama
import streamlit as st
import backend_processor

st.set_page_config(
    page_title="Diabetes in Pregnancy - Clinical Assistant",
    page_icon="🏥",
)

st.markdown(
    '<p style="font-family:sans-serif; color:Green; font-size: 36px;">'
    '🏥 Diabetes in Pregnancy - Clinical Assistant</p>',
    unsafe_allow_html=True,
)
st.caption("Guideline-based recommendations from NICE NG3: Diabetes in pregnancy")

# Load vector index once per session
if "vector_index" not in st.session_state:
    with st.spinner("📀 Loading and indexing the guideline..."):
        try:
            st.session_state.vector_index = backend_processor.get_index()
        except Exception as e:
            st.error(f"Failed to load guideline index: {e}")
            st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
input_text = st.chat_input("Enter your clinical query...")

if input_text:
    # Show user message
    st.chat_message("user").markdown(input_text)
    st.session_state.messages.append({"role": "user", "content": input_text})

    # Generate response with streaming
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching guideline..."):
            try:
                stream = backend_processor.hr_rag_response_stream(
                    index=st.session_state.vector_index, question=input_text
                )
            except Exception as e:
                st.error(f"Error searching guideline: {e}")
                stream = None

        if stream:
            try:
                response_content = st.write_stream(stream)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_content}
                )
            except Exception as e:
                st.error(f"Error generating response: {e}")
