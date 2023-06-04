from imutils import paths
import face_recognition
import pickle
import cv2
import os
import numpy as np
import io

imagePaths = list(paths.list_images('Images'))
knownEncodings = []
knownNames = []
for (i, imagePath) in enumerate(imagePaths):
    name = imagePath.split(os.path.sep)[-1]
    name =  os.path.splitext(name)[0]

    file_path_utf8 = f"Images/{name}.jpg"
    with io.open(file_path_utf8, 'rb') as f:
        image_data = f.read()
    nparr = np.frombuffer(image_data, np.uint8)

    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(original_image)
    boxes = face_recognition.face_locations(original_image,model='hog')
    encodings = face_recognition.face_encodings(original_image, boxes)

    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

data = {"encodings": knownEncodings, "names": knownNames}
f = open("face_enc", "wb")
f.write(pickle.dumps(data))
f.close()