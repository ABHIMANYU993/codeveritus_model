# ==============================================================================
# CODEVERITUS SYSTEM CORE MODULE
# ==============================================================================
# File: model/deployment/django_backend/dashboard/Results.py
# Author: ABHIMANYU993
# Email: abhimanyubadiger1001@gmail.com
# Project: Codeveritus - AI vs Human Code Classifier
# Description: Modular component containing system configuration or script details.
# ==============================================================================

import streamlit as st

# Results Page
st.title("Analysis Results")

# Check if there's a result from session state
if 'backend_result' in st.session_state:
    # Display the result from the backend
    st.write("Code Analysis Result:")
    st.json(st.session_state['backend_result'])
else:
    st.write("No result available. Please submit your code first.")