# app/ui_streamlit.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
st.set_page_config(page_title="Chatbot GenAI Plan de Desarrollo Institucional - UTP", layout="centered")

from app.rag_pipeline import load_vectorstore_from_disk, build_chain


st.title("🤖 Asistente del PDI - UTP")

question = st.text_input("Escribe tu pregunta sobre el plan de desarrollo institucional:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Cargar vectorstore y cadena
# ui_streamlit.py  (línea ~18)
vectordb = load_vectorstore_from_disk()
print("Vector DB type =", type(vectordb))
chain = build_chain(vectordb)

vectordb = load_vectorstore_from_disk()
chain = build_chain(vectordb)

if question:
    with st.spinner("Pensando..."):
        result = chain.invoke({"question": question, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.append((question, result["answer"]))

if st.session_state.chat_history:
    st.markdown("---")    
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**🧑 Usuario:** {q}")
        st.markdown(f"**🤖 Bot PDI:** {a}")
