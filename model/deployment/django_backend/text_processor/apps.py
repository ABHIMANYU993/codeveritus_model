# ==============================================================================
# CODEVERITUS SYSTEM CORE MODULE
# ==============================================================================
# File: model/deployment/django_backend/text_processor/apps.py
# Author: ABHIMANYU993
# Email: abhimanyubadiger1001@gmail.com
# Project: Codeveritus - AI vs Human Code Classifier
# Description: Modular component containing system configuration or script details.
# ==============================================================================

from django.apps import AppConfig


class TextProcessorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "text_processor"