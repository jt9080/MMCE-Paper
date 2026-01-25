import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.ai.textanalytics import TextAnalyticsClient

load_dotenv(override=True)

class AzureClient:
    def __init__(self):
        self.endpoint = os.getenv("VISION_ENDPOINT")
        self.key = os.getenv("VISION_KEY")
        if not self.endpoint or not self.key:
            raise ValueError("Azure Vision endpoint and key must be set in environment variables 'VISION_ENDPOINT' and 'VISION_KEY'.")
        self.client = ImageAnalysisClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key)
        )
        # Text Analytics client for NER
        self.language_key = os.getenv("LANGUAGE_KEY")
        self.language_endpoint = os.getenv("LANGUAGE_ENDPOINT")
        if not self.language_key or not self.language_endpoint:
            raise ValueError("Azure Language endpoint and key must be set in environment variables 'LANGUAGE_ENDPOINT' and 'LANGUAGE_KEY'.")
        self.text_analytics_client = TextAnalyticsClient(
            endpoint=self.language_endpoint,
            credential=AzureKeyCredential(self.language_key)
        )

    # Add abstraction methods for API calls here, e.g. analyze_image, etc.

    def get_image_details(self, image_path: str) -> dict:
        import os

        if not os.path.isfile(image_path):
            raise ValueError("image_path must be a valid local file path")

        with open(image_path, "rb") as image_file:
            result = self.client.analyze(
                image_file,
                visual_features=[
                    VisualFeatures.DENSE_CAPTIONS, 
                    VisualFeatures.READ, 
                    VisualFeatures.TAGS
                ]
            )
        details = {}
        # Dense Captions
        dense_captions = []
        if result.dense_captions is not None:
            for caption in result.dense_captions.list:
                dense_captions.append(caption.text)
        details["captions"] = dense_captions
        # OCR/Read (just text lines)
        ocr_lines = []
        if result.read is not None and result.read.blocks and len(result.read.blocks) > 0:
            for line in result.read.blocks[0].lines:
                for word in line.words:
                    ocr_lines.append(word.text)
        details["ocr"] = ocr_lines
        # Tags
        tags = []
        if result.tags is not None:
            for tag in result.tags.list:
                tags.append(tag.name)
        details["tags"] = tags
        return details

    def get_ner(self, documents):
        try:
            results = self.text_analytics_client.recognize_entities(documents=documents)
            all_entities = []
            for result in results:
                if not result.is_error:
                    for entity in result.entities:
                        all_entities.append(entity.text)
            return all_entities
        except Exception as err:
            print(f"Encountered exception during NER: {err}")
            return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AzureClient Test Runner")
    args = parser.parse_args()

    client = AzureClient()