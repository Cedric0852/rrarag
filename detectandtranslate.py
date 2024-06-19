from google.cloud import translate_v3 as translate
import google.api_core.exceptions

def detect_language(text: str, project_id: str = "banguka-406711") -> str:
    """Detecting the language of a text string.

    Args:
        text: The text to detect the language for.
        project_id: The GCP project ID.

    Returns:
        The detected language code.
    """
    client = translate.TranslationServiceClient()

    location = "global"
    parent = f"projects/{project_id}/locations/{location}"

    response = client.detect_language(
        content=text,
        parent=parent,
        mime_type="text/plain",  # mime types: text/plain, text/html
    )

    # Return the language code of the most probable language
    detected_language = response.languages[0].language_code
    return detected_language

async def translate_text(text: str, source_language: str, target_language: str, project_id: str = "banguka-406711") -> str:
    """Translating text from source language to target language.

    Args:
        text: The text to be translated.
        source_language: The source language code.
        target_language: The target language code.
        project_id: The GCP project ID.

    Returns:
        The translated text.
    """
    if source_language == target_language:
        return text  # No translation needed if source and target languages are the same

    client = translate.TranslationServiceClient()

    location = "global"
    parent = f"projects/{project_id}/locations/{location}"

    response = client.translate_text(
        parent=parent,
        contents=[text],
        mime_type="text/plain",
        source_language_code=source_language,
        target_language_code=target_language
    )

    # Return the translated text
    translated_text = response.translations[0].translated_text
    return translated_text