import cv2
import threading
import mediapipe as mp

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    
    def get_clean_frame(self):
        image = self.frame
        # _, jpeg = cv2.imencode('.jpg', image)
        return image

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()



            
