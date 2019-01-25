from keras.models import load_model
import cv2
import os
import glob
import numpy as np

model = load_model('models/facenet_keras.h5')
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

def euclidean_distance(image1, image2):
    distance = image1 - image2
    distance = np.sum(np.multiply(distance, distance))
    distance = np.sqrt(distance)
    return distance

def encoded_image(image):
    image = cv2.resize(image, (160, 160))
    image = np.around(image / 255.0, decimals = 12)
    x_train = np.array([image])
    return model.predict(x_train)

def prepare_database():
    database = {}
    for file in glob.glob("database/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        image = cv2.imread(file)

        face_image = extract_face_image(image)
        database[identity] = encoded_image(face_image)
    return database

def extract_face_image(image):
    (x1, y1, x2, y2) = extract_face_coordinates(image)[0]
    return sub_image(image, x1, y1, x2, y2)

def sub_image(image, x1, y1, x2, y2):
    height, width, channels = image.shape
    return image[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]

def extract_face_coordinates(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    all_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face_coordinates = []
    for (x1, y1, w, h) in all_faces:
        x2 = x1 + w
        y2 = y1 + h
        face_coordinates.append([x1, y1, x2, y2])

    return face_coordinates

def recognize_still_image(image):
    identities = []
    annotated_image = image.copy()
    for (x1, y1, x2, y2) in extract_face_coordinates(image):
        identity = find_identity(image, x1, y1, x2, y2)
        if identity is not None:
            annotated_image = annotate_image(annotated_image, identity, x1, y1, x2, y2)
            identities.append(identity)

    return annotated_image

def annotate_image(image, identity, x1, y1, x2, y2):
    image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, identity, (x1, y1), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return image

def find_identity(image, x1, y1, x2, y2):
    face_image = sub_image(image, x1, y1, x2, y2)
    encoding = encoded_image(face_image)

    min_dist = 100
    identity = None
    for (name, db_enc) in database.items():
        dist = euclidean_distance(db_enc, encoding)
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 9:
        return None
    else:
        print('Detected %s' %(identity))
        return str(identity)

if __name__ == "__main__":
    database = prepare_database()
    image_list = glob.glob('input/*')
    for (i, image_name) in enumerate(image_list):
        image = cv2.imread(image_name)
        output = recognize_still_image(image)
        cv2.imwrite('output/' + str(i) + '.jpg', output)
