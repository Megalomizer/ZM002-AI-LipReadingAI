import time

import cv2
import mediapipe


def main():
    capture = cv2.VideoCapture(0)

    min_detect_confidence = 0.7
    min_track_confidence = 0.7

    mp_holistic = mediapipe.solutions.holistic
    holistic_model = mp_holistic.Holistic(
        refine_face_landmarks=True,
        min_detection_confidence=min_detect_confidence,
        min_tracking_confidence=min_detect_confidence,
    )
    mp_drawing = mediapipe.solutions.drawing_utils

    previous_time = 0

    while True:
        # Capture frames
        has_captured, frame = capture.read()
        if not has_captured:
            continue

        # Flip frame
        frame = cv2.flip(frame, 1)

        # Convert to RGB to process frame with mediapipe for landmarks and revert to BGR
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame.flags.writeable = False
        results = holistic_model.process(processed_frame)
        processed_frame.flags.writeable = True
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

        # Draw the landmarks on the frame
        mp_drawing.draw_landmarks(
            processed_frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(
                color=(255, 0, 255),
                thickness=1,
                circle_radius=1,
            ),
            mp_drawing.DrawingSpec(
                color=(0, 255, 255),
                thickness=1,
                circle_radius=1,
            ),
        )

        # Grayscale the frame
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)

        # Crop the frame
        h, w = processed_frame.shape[:2]

        if results and results.face_landmarks:
            mouth_idx_set = set()
            for conn in mp_holistic.FACEMESH_CONTOURS:
                mouth_idx_set.update(conn)
            mouth_indices = sorted(mouth_idx_set)

            xs, ys = [], []
            for i in mouth_indices:
                lm = results.face_landmarks.landmark[i]
                x_px = int(lm.x * w)
                y_px = int(lm.y * h)
                xs.append(x_px); ys.append(y_px)

            if xs and ys:
                pad = 10
                x_min = max(0, min(xs) - pad); x_max = min(w, max(xs) + pad)
                y_min = max(0, min(ys) - pad); y_max = min(h, max(ys) + pad)

                cx = (x_min + x_max) // 2
                cy = (y_min + y_max) // 2

                side = max(x_max - x_min, y_max - y_min)
                half_side = side // 2

                x_min = max(0, cx - half_side); x_max = min(w, cx + half_side)
                y_min = max(0, cy - half_side); y_max = min(h, cy + half_side)

                processed_frame = processed_frame[y_min:y_max, x_min:x_max]
                processed_frame = cv2.resize(processed_frame, (500,500), interpolation=cv2.INTER_LANCZOS4)

        # Get FPS couter and add on top of frame for show
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(processed_frame, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        cv2.imshow("Original Frame", frame)
        cv2.imshow("Processed Frame", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()