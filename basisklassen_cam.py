import time
import cv2
import threading
import queue
import matplotlib.pyplot as plt

class Camera():
    """A class for the camera

    Args:
    skip_frame: number of frames to skip while recording
    cam_number: which camera should be used. Defaults to 0. 

    Attributes
    --------
    
    VideoCapture: Class to get the video feed
    _imgsize: size of the image

    Methods
    --------
    get_frame(): returns current frame, recorded by the camera
    show_frame(): plots the current frame, recorded by the camera
    get_jpeg(): returns the current frame as .jpeg/raw bytes file
    save_frame(): saves the frame at the given path under the given name
    release(): releases the camera, so it can be used again and by other programs
    """

    def __init__(self, skip_frame=2, cam_number=0, thread=False, fps=30):
        self.skip_frame = skip_frame
        self.VideoCapture = cv2.VideoCapture(cam_number, cv2.CAP_V4L) #,
        self.VideoCapture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.VideoCapture.set(cv2.CAP_PROP_FPS, fps)
        self._imgsize = (int(self.VideoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(self.VideoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if thread:
            self._queue_drive = queue.Queue(maxsize=1)
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    def _reader(self):
        """Put current frame recorded by the camera to the Queue-Object.
        """
        while True:

            frame = self.get_frame()

            if not self._queue_drive.empty():
                try:
                    self._queue_drive.get_nowait()
                except Queue.Empty:
                    pass
            self._queue_drive.put(frame)

    def read(self):
        """Returns current frame hold in Queue-Object.
        
        Returns:
            numpy array: returns current frame as numpy array
        """
        return self._queue_drive.get()
        
    def get_frame(self):
        """Returns current frame recorded by the camera

        Returns:
            numpy array: returns current frame as numpy array
        """
        if self.skip_frame:
            for i in range(int(self.skip_frame)):
                _, frame = self.VideoCapture.read()
        _, frame = self.VideoCapture.read()
        frame = cv2.flip(frame, -1)
        return frame

    def show_frame(self):
        """Plots the current frame
        """
        plt.imshow(self.get_frame())
    
    def get_jpeg(self, frame=None):
        """Returns the current frame as .jpeg/raw bytes file

        Args:
            frame (list): frame which should be saved.

        Returns:
            bytes: returns the frame as raw bytes
        """
        if frame is None:
            frame = self.get_frame()
        _,x = cv2.imencode('.jpeg', frame)
        return x.tobytes()

    def save_frame(self, path: str, name: str, frame=None):
        """Saves the current frame under the given path and filename

        Args:
            path (str): path where the file should be saved
            name (str): name under which the file should be saved
            frame (np.array, optional): frame which should be saved. 
                                        If None then the current frame recorded by the camera gets saved.
                                        Defaults to None.
        """
        if frame is None:
            frame = self.get_frame()
        cv2.imwrite(path + name, frame)

    def release(self):
        """Releases the camera so it can be used by other programs.
        """
        self.VideoCapture.release()

if __name__ == "__main__":
    cam = Camera(thread=True, fps=30)

    for i in range(20):

        start = time.perf_counter()

        frame = cam.read()
        print('Shape:', frame.shape)
        print('Time:', time.perf_counter()-start)