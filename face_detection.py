from tensorflow.keras import models
import tensorflow as tf
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

model = models.load_model('model/cnn_weights.h5', compile=False)
model.summary()
cv2.ocl.setUseOpenCL(False)
EMOTION_CLASSIFICATION = {0: "Angry", 1: "Disgust", 2: "Fear",
                          3: "Happy", 4: "Sad", 5: "Surprise", 6: 'Neutral'}
detector = MTCNN()
video_cap = cv2.VideoCapture(0)
black = np.zeros((96, 96))

while True:
    ok, frame = video_cap.read()
    if not ok:
        break
    res = detector.detect_faces(frame)
    if len(res) == 1:  # 0 if no face detected
        try:
            x1, y1, w, h = res[0]['box']
            x2, y2 = x1 + w, y1 + h
            face = frame[y1:y2, x1:x2]
            # rectangle boundary
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cropped = cv2.resize(face, (48, 48))
            cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            cropped_exp = np.expand_dims(cropped_gray, axis=0)
            cropped_float = cropped_exp.astype(float)
            prediction = model.predict(cropped_float)
            print(prediction)
            arg_max = int(np.argmax(prediction))
            cv2.putText(frame, EMOTION_CLASSIFICATION[arg_max], (
                x1+20, y1-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        except:
            pass
    cv2.imshow('Video', frame)
    try:
        cv2.imshow("frame", cropped)
    except:
        cv2.imshow("frame", black)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()
