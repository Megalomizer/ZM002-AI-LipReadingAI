import time
import threading
from queue import Queue, Empty
from core.constants import FRAMES_COLLECTION

from clients.video_client import VideoClient
from clients.database_client import DatabaseClient
from clients.detection_client import DetectionClient
from entities.lipreading_model import LipReadingModel
from clients.ollama_client import OllamaClient

def main():
    db_client = DatabaseClient()
    video_client = VideoClient()
    video_client.start()

    processing_interval = 0.5

    task_queue: Queue[list] = Queue(maxsize=1)
    stop_event = threading.Event()

    def processor_worker(q: Queue, stop_event: threading.Event):
        while not stop_event.is_set() or not q.empty():
            try:
                window = q.get(timeout=0.1)
            except Empty:
                continue

            try:
                # Heavy Duty loads
                print("\n-----------------------------------------------\nStarting task...")
                time.sleep(10)
                print("\nTask Completed!\n-----------------------------------------------")
            finally:
                q.task_done()

    def producer(q: Queue, stop_event: threading.Event):
        last_enqueued_frame = 0.0
        while not stop_event.is_set():
            now = time.time()
            if now - last_enqueued_frame >= processing_interval:
                last_enqueued_frame = now
                frames = video_client.get_buffer_window()
                if len(frames) >= FRAMES_COLLECTION:
                    window = frames[-FRAMES_COLLECTION:] # last x amount of frames as set by the constant
                    try:
                        # Blocking put — ensures no loss, may increase latency if worker is slower.
                        q.put(window, timeout=0.5)
                    except Exception:
                        # If we can't enqueue within timeout (e.g., shutdown), try again next loop.
                        pass
            # Small sleep to avoid busy-waiting
            time.sleep(0.001)

    worker = threading.Thread(target=processor_worker, args=(task_queue, stop_event), daemon=True)
    producer_thread = threading.Thread(target=producer, args=(task_queue, stop_event), daemon=True)
    worker.start()
    producer_thread.start()

    try:
        while True:
            video_client.show_latest_frame()

            if video_client.quit_requested():
                break
    finally:
        # Initiate shutdown
        # 1. Stop producer so no new tasks arrive
        stop_event.set()
        producer_thread.join(timeout=1.0)

        # 3. Wait for all queued tasks to finish to avoid data loss
        worker.join(timeout=1.0)

        # 4. Release resources
        video_client.release()

if __name__ == '__main__':
    main()