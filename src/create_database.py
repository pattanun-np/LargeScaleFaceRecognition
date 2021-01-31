import annoy,cv2
import pendulum
import json, os
import warnings
import config as CONFIG
from service.model.facenet_basemodel import FaceNet
from service.utils.prepare import preprocess_image
from service.components.facedector import face_detector
from tqdm.auto import tqdm
import numpy as np

class CreateDatabase:

    def __init__(self):
        self.model = FaceNet().loadModel(CONFIG.MODEL_PATH)
        self.facedetector = face_detector()

    def create(self, folder_path, dbpath):
        database = []
        data = {}
        aspect_size=(160,160)
        tree = annoy.AnnoyIndex(128, "angular")
        face_cord = {}
        for root, dirs, files in os.walk(folder_path):
            for i, file in tqdm(enumerate(files)):

                image_path = f"{root}/{file}"
                text = file.split('.')[0]
                name = text.split('_')[0]
                band = text.split('_')[-1]
                
                print(f"\nLoading: {image_path}")
                image = cv2.imread(image_path)
                face_pos, crop_face = self.facedetector.cropface(image)
                if len(face_pos[0]) == 4:
                    crop_face = np.expand_dims(crop_face,0)
                    embedding = self.model.predict(crop_face)[0, :]
                    vector = embedding.tolist()
                    tree.add_item(i, vector)
            
                    data = {
                        "id":f"{i}",
                        "name": f"{name}",
                        "band": f"{band}",
                        "image_path": f"{image_path}",
                        "update_time":f'{pendulum.now("Asia/Bangkok")}'
                        }
                    database.append(data)
                else:
                    print("Skip: {name}")
                    pass
            
        with open(f'{dbpath}', 'w',encoding="utf-8") as outfile:
            json.dump(database, outfile, ensure_ascii=False, indent=4)
        tree.build(10)
        tree.save('./service/database/database.tree')
        print("DONE")     
                 
if __name__ == '__main__':
    CreateDatabase().create(folder_path="./service/database/image",dbpath="./service/database/database.json")
        