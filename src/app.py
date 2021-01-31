import numpy as np
import cv2
from service.utils.prepare import preprocess_image
from service.utils.Thaitext import drawText
from service.components.facedector import face_detector
from service.model.facenet_basemodel import FaceNet
from service.database.database import Database
import config as CONFIG
import json

class main:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.face_detector = face_detector()
        self.model = FaceNet().loadModel(CONFIG.MODEL_PATH)
        self.database = Database(tree_path="./service/database/database.tree",database_path="./service/database/database.json")
        

    def run(self):
        while True:
            ret, cv_img = self.cap.read()
            for cv_img, faces, Is_capture, gray_img in self.face_detector.draw_bbox(cv_img):
                for i, (face, is_face,position) in enumerate(preprocess_image(cv_img, faces), 0):
                    if is_face:
                        face_encoded = self.model.predict(face)[0,:]
                        idx = self.database.match(face_encoded, n_of_similarity=1)
                        result, dist = self.database.getface(idx)
                        print(position[0])
                        cv_img = drawText(cv_img, text=f" Name: {result['name']} Band: {result['band']} Distance: {dist:.4f}", pos=position[0], fontSize=25, color=(255, 255, 255))                   
            cv2.imshow('img', cv_img)
            # Stop if escape key is pressed
            k = cv2.waitKey(30) & 0xff
            if k==27:
                break

        cap.release()

if __name__ == '__main__':
    main().run()