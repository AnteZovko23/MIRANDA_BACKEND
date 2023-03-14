import dlib
import cv2
import numpy as np
import pymongo
import bson

class Face_Recognition_Library:
    def __init__(self, DEBUG=False):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["miranda"]
        self.collection = self.db["face_features"]

        self.document = {
            "_id": bson.ObjectId(),
            "name": "",
            "embeddings": "",
            "landmarks": ""
        }

        self.model = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.DEBUG = DEBUG
        self.LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
        self.RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

    def insert_face(self, rect, img, name):
        landmarks = self.get_landmarks(img, rect)
        normalized_face = self.get_normalized_face(landmarks)
        embeddings = self.get_embeddings(normalized_face)

        self.document["embeddings"] = embeddings.tolist()
        self.document["landmarks"] = [(point.x, point.y) for point in landmarks.parts()]
        self.document["name"] = name

        self.collection.insert_one(self.document)

    def get_landmarks(self, img, rect):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return self.predictor(gray, rect)

    def get_embeddings(self, normalized_face):
        blob = cv2.dnn.blobFromImage(normalized_face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        self.model.setInput(blob)
        embeddings = self.model.forward()
        embeddings = embeddings[0]
        return embeddings

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def recognize_face(self, face_embeddings, target_embedding, threshold=0.5):
        best_match = None
        best_similarity = -1
        
        if self.DEBUG:
            face = self.collection.find_one({"name": "Ante"})
            target_embedding = np.array(face["embeddings"])
        
        similarity = self.cosine_similarity(face_embeddings, target_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = "Ante"
        if best_similarity < threshold:
            return "Not Ante"
        else:
            return "Ante"

    def get_normalized_face(self, img, landmarks):
        return dlib.get_face_chip(img, landmarks, size=96)


    def rect_to_tuple(self, rect):
        left = rect.left()
        right = rect.right()
        top = rect.top()
        bottom = rect.bottom()
        return left, top, right, bottom

    def extract_eye(self, shape, eye_indices):
        points = map(lambda i: shape.part(i), eye_indices)
        return list(points)

    def extract_eye_center(self, shape, eye_indices):
        points = self.extract_eye(shape, eye_indices)
        xs = map(lambda p: p.x, points)
        ys = map(lambda p: p.y, points)
        return sum(xs) // 6, sum(ys) // 6

    def extract_left_eye_center(self, shape):
        return self.extract_eye_center(shape, self.LEFT_EYE_INDICES)

    def extract_right_eye_center(self, shape):
        return self.extract_eye_center(shape, self.RIGHT_EYE_INDICES)

    def angle_between_2_points(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        tan = (y2 - y1) / (x2 - x1)
        return np.degrees(np.arctan(tan))

    def get_rotation_matrix(self, p1, p2):
        angle = self.angle_between_2_points(p1, p2)
        x1, y1 = p1
        x2, y2 = p2
        xc = (x1 + x2) // 2
        yc = (y1 + y2) // 2
        M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
        return M

    def crop_image(self, image, det):
        left, top, right, bottom = self.rect_to_tuple(det)
        return image[top:bottom, left:right]
