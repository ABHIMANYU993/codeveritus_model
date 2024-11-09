# ==============================================================================
# CODEVERITUS SYSTEM CORE MODULE
# ==============================================================================
# File: model/deployment/django_backend/dashboard/streamlit_app.py
# Author: ABHIMANYU993
# Email: abhimanyubadiger1001@gmail.com
# Project: Codeveritus - AI vs Human Code Classifier
# Description: Modular component containing system configuration or script details.
# ==============================================================================

import streamlit as st
import Home
import Results

# Simple navigation between the User page and Results page
if 'code_input' not in st.session_state:
    Home  # Display the user input page
else:
    Results  # Display the results page