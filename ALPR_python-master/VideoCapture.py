import threading
import cv2
import time
import json
import logging


class VideoCaptureThreading:
    def __init__(self, src=0, resolution=None, skip=None, wait=None, fps=None, fourcc=None):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.resolution = resolution
        self.skip = skip
        self.stopping = False
        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if resolution is not None:
            width, height = resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.fps = FPS()
        self.frames_grabbed = 0
        self.wait = wait
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Threaded video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            if self.wait:
                threading.Event().wait(self.wait)
            if self.skip:
                for i in range(self.skip):
                    self.cap.grab()
            grabbed, frame = self.cap.read()
            # fps = self.fps()
            if grabbed:
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
                    # print(f"Camera {self.src} thread fps: {fps:.2f}")
                    # logging.debug(f"Camera {self.src} thread fps: {fps:.2f}")
            else:
                # logging.warning("Camera thread Frame not grabbed!")
                self.stopping = True

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def generator(self):
        while True:
            with self.read_lock:
                if self.frame is None:
                    grabbed = False
                else:
                    frame = self.frame.copy()
                    grabbed = self.grabbed
            if grabbed:
                yield frame
            else:
                break

    def stop(self):
        self.started = False
        self.thread.join()
        self.cap.release()

    # def __exit__(self, exec_type, exc_value, traceback):
    #     self.cap.release()


class FPS:
    def __init__(self, decay=0.1):
        self.last_timestamp = time.time()
        self.decay = decay
        self.ema_fps = 0.0

    def __call__(self):
        ts = time.time()
        passed = ts - self.last_timestamp
        self.last_timestamp = ts
        self.ema_fps = 1./passed*self.decay+self.ema_fps*(1.-self.decay)
        return self.ema_fps


if __name__ == '__main__':

    st = time.time()
    with open('./config.json', 'rb') as f:
        print(type(json.load(f)))
    print("Taked time {}".format(time.time() - st))
