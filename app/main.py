from clients.video_client import VideoClient
from clients.detection_client import DetectionClient
from entities.lipreading_model import LipReadingModel
from clients.ollama_client import OllamaClient

def main():
    video_client = VideoClient()
    detection_client = DetectionClient()
    lipreading_model = LipReadingModel()
    ollama_client = OllamaClient()

    while True:
        frame = video_client.get_frame()
        if frame is None:
            break
        lips = detection_client.extract_lips(frame)
        if lips is not None:
            prediction = lipreading_model.predict(lips)
            refined = ollama_client.interpret(prediction)
            print(f"Prediction: {refined}\n")

        video_client.show_frame(frame)
        if video_client.quit_requested():
            break

    video_client.release()
    print("Hello World")

if __name__ == '__main__':
    main()