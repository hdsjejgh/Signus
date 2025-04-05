import mediapipe as mp
import os
import cv2 as cv

DATA_DIR = os.path.join("DATA_DIR")
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
FORMATTING = mp_drawing.DrawingSpec(color=(255,0,0),circle_radius=2,thickness=0)
SYMBOLS = ("a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","")
SYMBOL_MAPS = {c:i for i,c in enumerate(SYMBOLS)}
