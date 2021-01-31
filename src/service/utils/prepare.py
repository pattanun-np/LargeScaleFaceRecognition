import numpy as np
import cv2


def preprocess_image(image, face_pos, aspect_size=(160, 160)):
    is_face=False
    position=[]
    crop_faces = []
    if len(face_pos) > 0:
        for face in face_pos:
            is_face=True

            x = face[0],
            y = face[1],
            w = face[2],
            h = face[3],

            # print(image)
            crop_face = image[int(y[0]): int(h[0])+int(y[0]), int(x[0]): int(w[0])+int(x[0])]

            crop_face_h, crop_face_w, crop_face_ch  = crop_face.shape

            factorX, factorY, ch = crop_face.shape

            factorX = factorX/160
            factorY = factorY/160
            scaleX = 1 - factorX
            scaleY = 1 - factorY

            if scaleX or scaleY < 0:
                try:
                    image = cv2.resize(crop_face, dsize=(160, 160), interpolation=cv2.INTER_CUBIC)
                except:
                    is_face=False
                    pass
            else:
                try:
                    image = cv2.resize(crop_face, fx=scaleX, fy=scaleY, interpolation=cv2.INTER_CUBIC)
                except:
                    is_face=False
                    pass
                   
                
            image = np.expand_dims(image,0)
            crop_faces.append(image)
            position.append([int(x[0]),int(y[0])-70,int(w[0]),int(h[0])])
            
    else:
        print(f"No Faces")
        is_face=False
        pass
    
    yield  crop_faces,is_face,position



    