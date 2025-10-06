import time
import cv2
import mediapipe

from entities.camera_module import CameraModule


def main():
    camera_module = CameraModule()

    previous_time = 0
    no_captures_loop = 0

    while True:
        # Capture frames
        frame = camera_module.get_frame()
        if frame is None:
            no_captures_loop += 1
            if no_captures_loop >= 10:
                break
            continue
        original_frame = frame.copy()
        no_captures_loop = 0

        # Get landmarks
        landmark_detection_results = camera_module.get_landmarks(frame)

        # Continue only if landmarks are detected
        if landmark_detection_results.face_landmarks is not None:
            # Get lip coordinates
            lip_x_coords, lip_y_coords = camera_module.get_lip_coords(frame, landmark_detection_results)

            # Normalize the frame
            frame = camera_module.normalize_frame(frame, landmark_detection_results, lip_x_coords, lip_y_coords)

        # Get FPS couter and add on top of frame for show
        camera_module.set_fps_count(frame)

        cv2.imshow("Original Frame", original_frame)
        cv2.imshow("Processed Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera_module.release()

if __name__ == '__main__':
    main()