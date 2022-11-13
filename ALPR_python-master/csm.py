"""CLEANED"""
import threading
import cv2
import time
import logging

class VideoCaptureThreading:
    def __init__(self, src=0, resolution = None,skip=None, wait = None,fps=None,fourcc=None):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.resolution = resolution
        self.skip = skip
        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if resolution is not None:
            width,height = resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.frames_grabbed = 0
        self.wait = wait

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
            if grabbed:
                self.frames_grabbed+=1
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
                if self.frames_grabbed % 20 == 1:
                    pass
                    #logging.debug(f"Camera {self.src} thread fps: {fps:.2f}")
            else:
                logging.warning("Camera thread Frame not grabbed!")
                time.sleep(1)
    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame
    def generator(self):
        while True:
            with self.read_lock:
                frame = self.frame.copy()
                grabbed = self.grabbed
            if grabbed:
                yield frame
            else:
                break
    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()
