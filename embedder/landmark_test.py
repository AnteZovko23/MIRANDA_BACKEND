import dlib
import pymongo
import numpy as np
import cv2

# Load the shape predictor model
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Connect to the MongoDB database
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["miranda"]
face_features = db["face_features"]

# Load the original image
img = cv2.imread("ante_side.jpeg")

# Get the landmarks for the face
x, y, w, h = [int(i) for i in [338.0, 204.0, 622.0, 585.0]]
landmarks = predictor(img, dlib.rectangle(x, y, x + w, y + h))



# Convert the landmarks to a numpy array
landmarks_np = np.zeros((landmarks.num_parts, 2), dtype=np.int32)
for i in range(landmarks.num_parts):
    landmarks_np[i, 0] = landmarks.part(i).x
    landmarks_np[i, 1] = landmarks.part(i).y

# Query the database for existing landmarks
existing_landmarks = face_features.find_one({"name": "some guy"})

# Convert the existing landmarks to a numpy array
existing_landmarks = np.array(existing_landmarks["landmarks"], dtype=np.int32)

for i in range(landmarks.num_parts):
    cv2.circle(img, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 0, 255), -1)
    cv2.circle(img, (existing_landmarks[i, 0], existing_landmarks[i, 1]), 2, (0, 255, 0), -1)
    # Bounding box
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow("Transformed image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Check if the new landmarks are different enough from the existing landmarks
if not np.allclose(landmarks_np, existing_landmarks, atol=5):
    # Transform the landmarks using the existing landmarks
    M, _ = cv2.estimateAffine2D(landmarks_np, existing_landmarks)
    # Transform the landmarks using the existing landmarks
    

    # Apply the affine transformation to the full image
    img[y:y+h, x:x+w] = cv2.warpAffine(img[y:y+h, x:x+w], M, (w, h))

    # Update the database with the transformed landmarks
    # landmarks_collection.update_one({"landmarks": {"$exists": True}}, {"$set": {"landmarks": landmarks_np.tolist()}})

    # Show the transformed image
    # Draw the original landmarks and the transformed landmarks
    for i in range(landmarks.num_parts):
        cv2.circle(img, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 0, 255), -1)
        cv2.circle(img, (existing_landmarks[i, 0], existing_landmarks[i, 1]), 2, (0, 255, 0), -1)
    
    cv2.imshow("Transformed image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()