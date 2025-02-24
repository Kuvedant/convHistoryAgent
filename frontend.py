import streamlit as st
import requests
import json

# API URL
API_URL = "http://127.0.0.1:8000/query"  # Change this if using a remote API

st.set_page_config(page_title="AI Chat Agent", layout="centered")

st.title("🤖 AI Chat Agent with RAG")

# User Input
user_input = st.text_input("📝 Ask me anything:", "")

if st.button("Submit"):
    with st.spinner("Processing your request..."):
        response = requests.post(API_URL, json={"question": user_input})
        
        if response.status_code == 200:
            data = response.json()
            
            # Display Response
            st.subheader("💬 Response:")
            st.write(f"📢 {data['answer']}")
            
            # Display Retrieved Context in a more structured way
            st.subheader("📜 Retrieved Context:")
            if data["retrieved_context"]:
                for idx, context in enumerate(data["retrieved_context"], 1):
                    st.markdown(f"**{idx}.** `{context}`")
            else:
                st.write("No relevant context found.")

            # Display Classification
            st.subheader("📌 Classification:")
            st.write(f"🏷 {data['classification']}")
            
            # Display Verification Score
            st.subheader("✅ Verification Score:")
            st.progress(float(data["verification_score"]))
            st.write(f"Confidence: **{data['verification_score']:.2f}**")
        else:
            st.error("⚠️ Error communicating with API.")
