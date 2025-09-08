import cv2

class VideoClient:
    """
    Video client class
    -----------------------
    Handles Webcam input/output
    """

    def __init__(self, index = 0):
        self.cap = cv2.VideoCapture(index)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.flip(frame, 1) # Flip to mirrored view

    def show_frame(self):
        cv2.imshow('Lip Reading', self.get_frame())

    def quit_requested(self):
        return cv2.waitKey(1) & 0xFF == ord('q')

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()