#import antigravity
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import cv2 as cv
import mediapipe as mp
from mp_functions import *
from const import *
import numpy as np
import tensorflow as tf
from collections import deque
import pyautogui

capture = cv.VideoCapture(0)
model = tf.keras.models.load_model('model2.keras')
buffer = deque()
lastSymb = None
NoneCount = 0

def act(action):
    if action in SYMBOLS[:36]:
        pyautogui.press(action)
        pyautogui.keyUp('alt')
        pyautogui.keyUp('ctrl')
        pyautogui.keyUp('shift')
    elif action.lower() in ('alt','shift','ctrl','space','esc','enter','backspace'):
        pyautogui.keyDown(action.lower())
    elif action == 'Caps':
        pyautogui.keyDown('capslock')
    elif SYMB_CONV.get(action,None) is not None:
        pyautogui.press(SYMB_CONV[action])
    else:
        if action =='M3up':
            pyautogui.scroll(5)
        elif action =='M3down':
            pyautogui.scroll(-5)
        elif action =='M1':
            pyautogui.click()
        elif action =='M2':
            pyautogui.click(button='right')
        elif action == 'Mup':
            pyautogui.move(0,-50)
        elif action == 'Mdown':
            pyautogui.move(0,50)
        elif action == 'Mleft':
            pyautogui.move(-50,0)
        elif action == 'Mright':
            pyautogui.move(50,0)


with mp_holistic.Holistic(min_tracking_confidence=0.5,min_detection_confidence=0.5) as holistic:
    while True:
        ret, frame = capture.read()

        results = mp_detection(frame,holistic)
        mp_draw(frame,results)


        if len(buffer)==15:
            if results.left_hand_landmarks or results.right_hand_landmarks:
                buffer.popleft()
                buffer.append(extract(results))
                #print(extract(results))

                pred = model.predict(np.expand_dims(np.array(buffer),axis=0),verbose=0)
                if np.max(pred)>=THRESHOLD:
                    if lastSymb != INV_SYMBOLS[np.argmax(pred)]:
                        print(f"Thats probably {INV_SYMBOLS[np.argmax(pred)]}. {np.max(pred)*100}% sure")
                        act(INV_SYMBOLS[np.argmax(pred)])
                        NoneCount=0
                    lastSymb = INV_SYMBOLS[np.argmax(pred)]
            else:
                if lastSymb is not None:
                    print(None)
                    NoneCount+=1
                    if NoneCount>8:
                        lastSymb=None
                        print('Noned')
                        buffer.clear()

        else:
            if results.left_hand_landmarks or results.right_hand_landmarks:
                buffer.append(extract(results))
        cv.imshow('Feed', frame)


        if cv.waitKey(20)&0xFF == ord('x'):
            break
