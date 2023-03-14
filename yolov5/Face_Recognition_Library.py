import dlib
import cv2
import numpy as np
import pymongo
import bson
from scipy.spatial.distance import mahalanobis
from facenet_pytorch import InceptionResnetV1
import torch

class Face_Recognition_Library:
    def __init__(self, DEBUG=False):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["miranda"]
        self.collection = self.db["face_features"]

        self.document = {
            "_id": bson.ObjectId(),
            "name": "",
            "embeddings": "",
            "voice_file": "",
        }

        self.model = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")
        # Load the VGG_Face.t7 pretrained model
        # self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.DEBUG = DEBUG
        self.LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
        self.RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

    def insert_face(self, name, embeddings=None, FILE=None):
        self.document["embeddings"] = embeddings.tolist()
        # self.document["landmarks"] = [(point.x, point.y) for point in landmarks.parts()]
        self.document["name"] = name
        
        self.document["voice_file"] = FILE
        self.collection.insert_one(self.document)
        print("Inserted face")

    def get_landmarks(self, img, rect):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        temp = self.predictor(gray, rect)
        return temp
        

    def get_embeddings(self, normalized_face):
        blob = cv2.dnn.blobFromImage(normalized_face, 1.0 / 255, (96,96), (0, 0, 0), swapRB=True, crop=False)
        self.model.setInput(blob)
        embeddings = self.model.forward()
        embeddings = embeddings[0]
        return embeddings

        # face = cv2.cvtColor(normalized_face, cv2.COLOR_BGR2RGB)
        # face = np.transpose(face, (2,0,1)).astype(np.float32)
        # face = np.expand_dims(face, axis=0)
        # embeddings = self.model(torch.from_numpy(face))
        # return embeddings.detach().numpy()[0]



    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    
    def l2_norm_cosine_similarity(self, face_embedding, target_embedding):
        # L2-normalize the face and target embeddings
        face_embedding = face_embedding / np.linalg.norm(face_embedding)
        target_embedding = target_embedding / np.linalg.norm(target_embedding)
        
        # Calculate the dot product between the normalized embeddings
        dot_product = np.dot(face_embedding, target_embedding)
        
        # Calculate the L2-normalized cosine similarity
        similarity = 0.5 * (dot_product + 1)
        
        return similarity
    
    def l2_normalized_cosine_similarity(self, embedding1, embedding2):
        # Calculate the dot product between the two vectors
        dot_product = np.dot(embedding1, embedding2)
        
        # Calculate the L2 norm (i.e., Euclidean length) of each vector
        embedding1_norm = np.linalg.norm(embedding1)
        embedding2_norm = np.linalg.norm(embedding2)
        
        # Calculate the cosine similarity and L2 normalize the result
        similarity = dot_product / (embedding1_norm * embedding2_norm)
        return similarity
        


    def recognize_face(self, face_embeddings, target_embedding, threshold=0.5):
        best_match = "unknown"
        best_similarity = -1
        voice_rec = None
        
        faces = list(self.collection.find({}))
        for face in faces:
            
            face_embeddings = np.array(face["embeddings"])
            similarity = self.l2_normalized_cosine_similarity(face_embeddings, target_embedding)
        
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = face["name"]
                voice_rec = face["voice_file"]
                
        if best_similarity < threshold:
            return "unknown", voice_rec
        else:
            return best_match, voice_rec
     
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
 
        
    def get_normalized_face(self, img, rect):
        scale = 2
        height, width = img.shape[:2]
        s_height, s_width = height // scale, width // scale
        img = cv2.resize(img, (s_width, s_height))
        # Turn rectangle into rectangles for dlib
        # Resize rect
        rect = dlib.rectangle(int(rect.left() / scale), int(rect.top() / scale), int(rect.right() / scale), int(rect.bottom() / scale))
        
        dets = dlib.rectangles()
        
        dets.append(rect)
        
        for i, det in enumerate(dets):
            shape = self.predictor(img, det)
            left_eye = self.extract_left_eye_center(shape)
            right_eye = self.extract_right_eye_center(shape)

            M = self.get_rotation_matrix(left_eye, right_eye)
            rotated = cv2.warpAffine(img, M, (s_width, s_height), flags=cv2.INTER_CUBIC)

            cropped = self.crop_image(rotated, det)
            
            return cropped