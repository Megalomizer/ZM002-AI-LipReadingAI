import numpy
from core.constants import VISUAL_MODEL
from tensorflow import keras
from tensorflow.keras.models import load_model
from transformers import AutoModelForCTC, AutoFeatureExtractor, AutoTokenizer

class LipReadingModel:
    """
    The wrapper for the trained lip-reading model
    """
    def __init__(self):
        self.extractor = AutoFeatureExtractor.from_pretrained(VISUAL_MODEL)
        self.model = AutoModelForCTC.from_pretrained(VISUAL_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(VISUAL_MODEL)

    def predict(self, lips):
        # TODO: preprocess lips & run through model
        # return "Hello World!"
        inputs = self.extractor(lips, return_tensors="pt") # Batch processing
        logits = self.model(**inputs).logits
        pred_ids = logits.argmax(dim=-1)
        return self.tokenizer.batch_decode(pred_ids)[0]

    def _preprocess_frames_to_lip_crops(self):
        """
        Convert raw BGR frames (from cv2) into cropped mouth regions suitable for the feature extractor.

        - frames: List of BGR frames from cv2
        - detection_client: object providing extract_lips(frame) -> list of (x, y) landmark points or None
        - crop_size: resize target for mouth crops (w, h)
        - margin: extra pixels added around the lips bounding box

        :return:
        """