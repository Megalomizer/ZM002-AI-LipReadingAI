import collections
import threading
import time

import cv2
from core.constants import FRAMES_COLLECTION

class VideoClient:
    """
    Video client class
    -----------------------
    Handles Webcam input/output.

    Usage:
    - Call start(target_fps=30) once to begin background capture.
    - In your UI loop, call show_latest_frame() each iteration to render the
      most recent frame and quit_requested() to handle user exit.
    - Use get_latest_frame() when you need to read the newest frame for processing.
    - Use get_buffer_window() to obtain a snapshot of the recent frames buffer
      (size controlled by FRAMES_COLLECTION).
    - Call release() on shutdown to free camera and UI resources.

    Thread-safety:
    - Latest frame and buffer are protected by an internal lock.
    - get_latest_frame() and get_buffer_window() return copies/snapshots safe to use
      outside the lock.
    """


    def __init__(self, index = 0):
        """
        Initialize the video capture client.

        Params:
        - index: int (default=0)
          The camera index passed to cv2.VideoCapture. Use 0 for the default webcam.

        Notes:
        - Attempts to reduce internal buffering via CAP_PROP_BUFFERSIZE to minimize latency.
        - No capture thread is started until start() is called.
        """

        cv2.setUseOptimized(True)
        self.cap = cv2.VideoCapture(index)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._latest = None
        self._buffer = collections.deque(maxlen=FRAMES_COLLECTION)

    def _get_frame(self):
        """
        Grab one frame from the camera, flipped horizontally for a mirrored view.

        Returns:
        - frame: numpy.ndarray (BGR) if successful
        - None if the capture fails or no frame is available
        """

        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.flip(frame, 1)
    
    def start(self, target_fps=None):
        """
        Start the background capture thread.

        Params:
        - target_fps: Optional[int]
          If provided, the capture loop will sleep to approximately match this FPS.
          If None, it will run as fast as possible (subject to hardware and backend limits).

        Notes:
        - Safe to call multiple times; subsequent calls are no-ops if already running.
        """

        if self._running:
            return
        # Validate camera before starting the capture loop
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Failed to open camera. Check camera availability and permissions.")
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, args=(target_fps,), daemon=True)
        self._thread.start()
        
    def _capture_loop(self, target_fps):
        """
        Internal background capture loop.

        Params:
        - target_fps: Optional[int]
          Desired capture rate; used to compute sleep time between reads.

        Behavior:
        - Continuously grabs frames, updates the latest frame, and appends to an internal
          bounded buffer (deque) capped by FRAMES_COLLECTION.
        - If a frame cannot be read, it briefly sleeps and tries again.
        - Runs until _running is set to False.
        """

        sleep_time = 1.0 / target_fps if target_fps else 0.0
        while self._running:
            frame = self._get_frame()
            # If we fail to get a frame, back off briefly and retry
            if frame is None:
                time.sleep(0.005)
                continue
            # On a valid frame, update latest and buffer under lock
            with self._lock:
                self._latest = frame
                self._buffer.append(frame)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    def get_latest_frame(self):
        """
        Get a copy of the most recent frame.

        Returns:
        - numpy.ndarray (BGR) copy of the latest frame, or None if no frame is available.

        Thread-safety:
        - Returns a copy under lock to avoid race conditions with the capture thread.
        """

        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def get_buffer_window(self):
        """
        Get a snapshot of the recent frames buffer.

        Returns:
        - List[numpy.ndarray] of frames (BGR), ordered from oldest to newest.
          The length is at most FRAMES_COLLECTION.

        Use cases:
        - Batch/sequence processing (e.g., lipreading over a sliding window).
        """

        with self._lock:
            return list(self._buffer)
        
    def show_latest_frame(self):
        """
        Display the most recent frame in a window titled 'LipReading'.

        Notes:
        - Call cv2.waitKey(1) regularly (e.g., via quit_requested()) to keep the UI responsive.
        - Does nothing if no frame has been captured yet.
        """

        frame = self.get_latest_frame()
        if frame is not None:
            cv2.imshow("LipReading", frame)

    def quit_requested(self):
        """
        Check for user request to quit the application.

        Returns:
        - True if the 'q' key was pressed, False otherwise.

        Notes:
        - This also gives time to the UI event loop by calling cv2.waitKey(1).
        - Call this every iteration of your main loop to keep the window responsive.
        """

        return cv2.waitKey(1) & 0xFF == ord('q')

    def release(self):
        """
        Release camera and UI resources.

        Behavior:
        - Stops the capture (if running) and joins the thread.
        - Releases the cv2.VideoCapture and destroys OpenCV windows.

        Call this once on shutdown to avoid resource leaks.
        """

        self.cap.release()
        cv2.destroyAllWindows()