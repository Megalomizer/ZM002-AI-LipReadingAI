import cv2
import mediapipe as mp

class DetectionClient:
    """
    Lip detector client class
    -----------------------
    Handles the detection of the Lips
    """
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    def extract_lips(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            lips = [
                (int(pt.x * w), int(pt.y * h))
                for lm in results.multi_face_landmarks
                for i, pt in enumerate(lm.landmark) if 61 <= i <= 90
            ]
            return lips
        return None
