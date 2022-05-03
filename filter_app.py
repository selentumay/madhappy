from ast import arg
from concurrent.futures import thread
import tensorflow as tf
import cv2
import numpy as np
import threading
from cnn import generateModel


model = generateModel()
model.load_weights('/Users/selentumay/cs1430/madhappy/data/checkpoints/your.weights.e011-acc0.8602.h5')
model.summary()

cv2.ocl.setUseOpenCL(False)

EMOTION_CLASSIFICATION = {0: 'Happy', 1: 'Sad', 2: 'Surprise'} #{0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: 'Neutral'}
video_cap = cv2.VideoCapture(0)
emotion_ind = 0


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def apply_mask(face: np.array, mask: np.array) -> np.array:
    mask_h, mask_w, _ = mask.shape
    face_h, face_w, _ = face.shape

    # Resize the mask to fit on face
    factor = min(face_h / mask_h, face_w / mask_w)
    new_mask_w = int(factor * mask_w)
    new_mask_h = int(factor * mask_h)
    new_mask_shape = (new_mask_w, new_mask_h)
    resized_mask = cv2.resize(mask, new_mask_shape)

    face_with_mask = face.copy()
    non_white_pixels = (resized_mask < 230).all(axis=2)
    off_h = int((face_h - new_mask_h) / 2)
    off_w = int((face_w - new_mask_w) / 2)
    face_with_mask[off_h: off_h+new_mask_h, off_w: off_w+new_mask_w][non_white_pixels] = resized_mask[non_white_pixels]

    return face_with_mask


def check_emotion(frame, res):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4
    )
    if len(faces) > 0:
        (x1, y1, w, h) = faces[0]
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        
        face = gray[y1:y2, x1:x2]
        #rectangle boundary
        cropped = cv2.resize(face, (48,48)) 

        cropped_exp = np.expand_dims(cropped, axis=0)
        cropped_float = cropped_exp.astype(float)

        prediction = model.predict(cropped_float)
        #print(prediction)

        numzeros = 0

        #print(prediction)


        if np.count_nonzero(prediction) == 1:
            
            emotion_ind = int(np.argmax(prediction))

            result.append(EMOTION_CLASSIFICATION[emotion_ind])

        else:
            result.append('Neutral')
            
        return

ok, frame = video_cap.read()
result =[""]
t = threading.Thread(target=check_emotion, args=(frame, result))
t.start()
show_emotion = False
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cowboy = cv2.imread('filters/cowboy3.png')
dog = cv2.imread('filters/dog.png')
empty = cv2.imread('filters/empty.png')
happy = cv2.imread('filters/beaches.png')

while True:
    ok, frame = video_cap.read()
    frame_h, frame_w, _ = frame.shape
    
    if not(t.is_alive()):
        t.join()

        t = threading.Thread(target=check_emotion, args=(frame, result))
        t.start()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4
    )

    if len(faces) > 0:
        (x1, y1, w, h) = faces[0]
        x2, y2 = x1 + w, y1 + h
        rect = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        offset = -130
        mask = empty
        if result[-1] == 'Sad':
            mask = dog
        if result[-1] == 'Surprise':    
            mask = cowboy
        if result[-1] == 'Happy':    
            mask = happy
        if x1<0 or y1+offset<0 or x2> frame_w or y2>frame_h:
            continue
        
        frame[y1+offset:y2, x1:x2] = apply_mask(frame[y1+offset:y2, x1:x2], mask)
    cv2.putText(frame, result[-1], (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Video',frame)
        

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()