from google.cloud import translate_v3 as translate
import google.api_core.exceptions
import json
import streamlit as st
import google.auth
from google.oauth2 import service_account
def get_translate_client():
    gcp_credentials = st.secrets["gcp_service_account"]
    credentials_dict = json.loads(json.dumps(dict(gcp_credentials)))
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)
    
    return translate.TranslationServiceClient(credentials=credentials)
def detect_language(text: str, project_id: str = "banguka-406711") -> str:
    client = get_translate_client()

    project_id = st.secrets["gcp_service_account"]["project_id"]

    location = "global"
    parent = f"projects/{project_id}/locations/{location}"

    response = client.detect_language(
        content=text,
        parent=parent,
        mime_type="text/plain",  
    )
    detected_language = response.languages[0].language_code
    return detected_language

async def translate_text(text: str, source_language: str, target_language: str, project_id: str = "banguka-406711") -> str:
    if source_language == target_language:
        return text  
    client = get_translate_client()
    project_id = st.secrets["gcp_service_account"]["project_id"]
    location = "global"
    parent = f"projects/{project_id}/locations/{location}"

    response = client.translate_text(
        parent=parent,
        contents=[text],
        mime_type="text/plain",
        source_language_code=source_language,
        target_language_code=target_language
    )
    translated_text = response.translations[0].translated_text
    return translated_text