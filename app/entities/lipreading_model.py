from core.constants import VISUAL_MODEL

class LipReadingModel:
    """
    The wrapper for the trained lip-reading model
    """
    def __init__(self):
        # TODO: Load pretrained/selftrained model like Lipnet or AV-HuBERT
        self.model = VISUAL_MODEL

    def predict(self, lips):
        # TODO: preprocess lips & run through model
        return "Hello World!"