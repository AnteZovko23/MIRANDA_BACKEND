import dlib
import cv2
import numpy as np
import pymongo
import bson
# Open the database connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["miranda"]
collection = db["face_features"]

# Create a dictionary to be inserted as a document in the database
document = {
    "_id": bson.ObjectId(),
    "name": "",
    "embeddings": "",
    "landmarks": ""

}

# Load the pre-trained model
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
model = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")

# Load the image
img = cv2.imread("example.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
detector = dlib.get_frontal_face_detector()
faces = detector(gray, 1)

# Load the pre-trained model

# Get the height and width of the image
h, w = img.shape[:2]

# Create a blob from the image and set the input to the model
blob = cv2.dnn.blobFromImage(img, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
model.setInput(blob)

# Run a forward pass through the model to get the embeddings and landmarks
vec = model.forward()

# Flatten the embeddings to a 1D array
embeddings = vec.flatten()
# np.save("embeddings", embeddings)
# Save embeddings as a list
embeddings_list = embeddings.tolist()
document["embeddings"] = embeddings_list


print("Embeddings:", embeddings)
landmarks_list = []
# Loop over the faces
# for face in faces:
#     # Get the landmarks for the face
#     landmarks = predictor(gray, face)
    
#     # Loop over the landmarks and draw them on the image
#     for i in range(0, 68):
#         x = landmarks.part(i).x
#         y = landmarks.part(i).y
#         cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
#         landmarks_list.append((x, y))

# Show the image with landmarks
# np.save("landmarks", landmarks)
document["landmarks"] = landmarks_list
# Save name
# np.save("name", "example")
document["name"] = "some guy"
print("Document:", document)
collection.insert_one(document)


# cv2.imshow("Landmarks", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



