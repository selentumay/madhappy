from ast import arg
from concurrent.futures import thread
from itertools import count
import tensorflow as tf
import cv2
import numpy as np
import threading
from cnn import generateModel


model = generateModel()
model.load_weights('/Users/giacomomarino/cs1430/madhappy/data/checkpoints/your.weights.e017-acc0.8708.h5')
model.summary()

cv2.ocl.setUseOpenCL(False)

EMOTION_CLASSIFICATION = {0: 'Happy', 1: 'Sad', 2: 'Surprise'} #{0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: 'Neutral'}
video_cap = cv2.VideoCapture(0)


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

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
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        
        face = gray[y1:y2, x1:x2]
        #rectangle boundary
        cropped = cv2.resize(face, (48,48)) 

        cropped_exp = np.expand_dims(cropped, axis=0)
        cropped_float = cropped_exp.astype(float)

        prediction = model.predict(cropped_float)

        if np.count_nonzero(prediction) == 1:
            
            i = int(np.argmax(prediction))

            result.append(EMOTION_CLASSIFICATION[i])

        else:
            result.append('Neutral')
            
        return

ok, frame = video_cap.read()
result =["Neutral", "Neutral", "Neutral"]
t = threading.Thread(target=check_emotion, args=(frame, result))
t.start()
show_emotion = False
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ok, frame = video_cap.read()
    
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

    if len(result) > 10:
        result = result[-8:]


    find_emot = {em:result[-3:].count(em) for em in result[-3:]}
    if 'Sad' in find_emot:
        find_emot = 'Sad'
    else: find_emot = sorted(find_emot, key = lambda t: t[1])[-1]

    if len(faces) > 0:
        (x1, y1, w, h) = faces[0]
        x2, y2 = x1 + w, y1 + h
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, find_emot, (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Video',frame)
        

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()