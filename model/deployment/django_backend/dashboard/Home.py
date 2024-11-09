import streamlit as st
import requests

# User Page for Input
st.title("Code Input Page")

# Text area to accept code input
code_input = st.text_area("Enter your code here:", height=200)

# Button to submit code
if st.button("Submit Code"):
    # Save the input to session state
    st.session_state['code_input'] = code_input

    # Send code to the backend (API call)
    try:
        response = requests.post("http://localhost:8000/analyze_code/", json={"code": code_input})
        # Store the result from backend in session state
        st.session_state['backend_result'] = response.json().get("result")

        # Redirect to results page
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Failed to connect to backend: {e}")