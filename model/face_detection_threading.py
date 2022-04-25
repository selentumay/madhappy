from ast import arg
from concurrent.futures import thread
from tensorflow.keras import models
import tensorflow as tf
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import threading

model = models.load_model('model/cnn_weights.h5', compile=False)
model.summary()
cv2.ocl.setUseOpenCL(False)
EMOTION_CLASSIFICATION = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: 'Neutral'}
detector = MTCNN()
video_cap = cv2.VideoCapture(0)
black = np.zeros((96,96))


def check_emotion(frame, res):
    res = detector.detect_faces(frame)
    if len(res) == 1: #0 if no face detected
        try:
            x1, y1, w, h = res[0]['box']
            x2, y2 = x1 + w, y1 + h
            face = frame[y1:y2, x1:x2]
            #rectangle boundary
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cropped = cv2.resize(face, (48,48)) 
            cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            cropped_exp = np.expand_dims(cropped_gray, axis=0)
            cropped_float = cropped_exp.astype(float)

            prediction = model.predict(cropped_float)
            i = int(np.argmax(prediction))
            print(prediction)
            print("appending emotion", EMOTION_CLASSIFICATION[i])
            print("")
            result.append(EMOTION_CLASSIFICATION[i])
            return
        except:
            pass

ok, frame = video_cap.read()
result =[""]
t = threading.Thread(target=check_emotion, args=(frame, result))
t.start()
show_emotion = False

while True:
    ok, frame = video_cap.read()
    
    if not(t.is_alive()):
        print(result)
  
        t.join()

        t = threading.Thread(target=check_emotion, args=(frame, result))
        t.start()

    cv2.putText(frame, result[-1], (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Video',frame)
        

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()