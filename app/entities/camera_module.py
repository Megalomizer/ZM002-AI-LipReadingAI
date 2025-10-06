import time
import cv2
import mediapipe

class CameraModule:
    IMAGE_SIZE = 500
    def __init__(self, camera_id = 0, min_detection_confidence = 0.7, min_track_confidence = 0.7):
        self.capture = cv2.VideoCapture(camera_id)
        self.mp_holistic = mediapipe.solutions.holistic
        self.holistic_model = self.mp_holistic.Holistic(
            refine_face_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_track_confidence,
        )
        self.mp_drawing = mediapipe.solutions.drawing_utils
        self.time = time.time()

    def get_frame(self):
        has_captured, frame = self.capture.read()
        if not has_captured:
            return None
        return cv2.flip(frame, 1)

    def set_fps_count(self, frame):
        new_time = time.time()
        fps = 1 / (new_time - self.time)
        self.time = time
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    def release(self):
        self.capture.release()
        cv2.destroyAllWindows()

    def get_landmarks(self, frame):
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame.flags.writeable = False
        return self.holistic_model.process(processed_frame)

    def get_lip_coords(self, frame, landmark_results):
        h, w = frame.shape[:2]

        inner_lip_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
        outer_lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        lip_indices = sorted(inner_lip_indices + outer_lip_indices)

        xs, ys = [], []
        for i in lip_indices:
            lm = landmark_results.face_landmarks.landmark[i]
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)
            xs.append(x_px)
            ys.append(y_px)

        return xs, ys

    def normalize_frame(self, frame, landmarks, lip_x_coords, lip_y_coords):
        # Draw landmarks
        self._draw_landmarks(frame, landmarks)

        # Grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Crop
        frame = self._crop_frame(frame, lip_x_coords, lip_y_coords)

        return frame

    def _draw_landmarks(self, frame, landmarks):
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks.face_landmarks,
            self.mp_holistic.FACEMESH_CONTOURS,
            self.mp_drawing.DrawingSpec(
                color=(255, 0, 255),
                thickness=1,
                circle_radius=1,
            ),
            self.mp_drawing.DrawingSpec(
                color=(0, 255, 255),
                thickness=1,
                circle_radius=1,
            ),
        )

    def _crop_frame(self, frame, lip_x_coords, lip_y_coords):
        h, w = frame.shape[:2]
        pad = 12

        x_min = max(0, min(lip_x_coords) - pad); x_max = min(w, max(lip_x_coords) + pad)
        y_min = max(0, min(lip_y_coords) - pad); y_max = min(h, max(lip_y_coords) + pad)

        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        half_side = max(x_max - x_min, y_max - y_min) // 2 # Squared image locked

        x_min = max(0, cx - half_side); x_max = min(w, cx + half_side)
        y_min = max(0, cy - half_side); y_max = min(h, cy + half_side)
        frame = frame[y_min:y_max, x_min:x_max]
        return cv2.resize(frame, (self.IMAGE_SIZE, self.IMAGE_SIZE), interpolation=cv2.INTER_LANCZOS4)