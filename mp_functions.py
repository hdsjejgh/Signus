import cv2 as cv
from const import *
import numpy as np

def mp_detection(img, model):
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = model.process(img)
    img.flags.writeable = True
    #img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    return results

def mp_draw(img,results):
    mp_drawing.draw_landmarks(img,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS, FORMATTING)
    mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, FORMATTING)
    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, FORMATTING)

def extract(results):
    pose = np.array([[res.x, res.y, res.x, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros((132,))
    rh = np.array([[res.x, res.y, res.x] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros((63,))
    lh = np.array([[res.x, res.y, res.x] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros((63,))
    return np.concatenate([lh, rh])
