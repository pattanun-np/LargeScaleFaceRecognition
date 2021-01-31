import annoy
import numpy as np
import os
import time
import cv2
import json

class Database:

    def __init__(self, tree_path, database_path):

        self.model = annoy.AnnoyIndex(128, "angular")
        self.model.load(tree_path)

        with open(database_path, "r") as f:
            self.database = json.load(f)

    def match(self, face_encoded, n_of_similarity=1, include_distances=True):
        tic = time.clock()
        idx = self.model.get_nns_by_vector(face_encoded, n_of_similarity, include_distances=True)
        toc = time.clock()
        used_time = toc - tic
        return idx

    def getface(self, idx, theshold=0.5):
        result = {}
        for matched, dist in zip(idx[0], idx[1]):
            # print(matched,dist)
            if dist <= theshold:
                result = self.database[matched]
            else:
                result = {
                        "index":-1,
                        "name": f"unknown",
                        "band":"unknown",
                        "image_path":f"unknown",
                        "added_time": f"unknown"
                    }
        return result,dist
