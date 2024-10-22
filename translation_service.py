import os
import logging
import requests
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TranslationRequest(BaseModel):
    src: str
    tgt: str
    use_multi: str
    text: str
    
    class Config:
        populate_by_name = True

class TranslationService:
    def __init__(self):
        self.api_url = os.getenv('TRANSLATION_API_URL')
        if not self.api_url:
            raise ValueError("TRANSLATION_API_URL environment variable is not set")

    def translate(self, text: str, src_language: str, tgt_language: str) -> str:
        try:
            payload = TranslationRequest(
                src=src_language,
                tgt=tgt_language,
                use_multi="MULTI",
                text=text
            )

            response = requests.post(
                self.api_url,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json"
                },
                json=payload.model_dump()
            )

            if response.status_code == 200:
                return response.json().get("translation")
            elif response.status_code == 406:
                raise ValueError("Invalid language pair selected")
            else:
                raise ValueError(f"Translation failed with status code {response.status_code}")
        except Exception as e:
            logging.error(f"Translation error: {str(e)}")
            return text