# ==============================================================================
# CODEVERITUS SYSTEM CORE MODULE
# ==============================================================================
# File: model/deployment/django_backend/text_processor/urls.py
# Author: ABHIMANYU993
# Email: abhimanyubadiger1001@gmail.com
# Project: Codeveritus - AI vs Human Code Classifier
# Description: Modular component containing system configuration or script details.
# ==============================================================================

from django.urls import path
from .views import process_text

urlpatterns = [
    path('process-text/', process_text, name='process_text'),
]