import mediapipe as mp
import os
import cv2 as cv
import numpy as np

DATA_DIR = os.path.join("DATA_DIR")
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
FORMATTING = mp_drawing.DrawingSpec(color=(255,0,0),circle_radius=2,thickness=0)
FRAMES = 15
VID_NUM = 60
SYMBOLS = np.array(("a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z",
           "0","1","2","3","4","5","6","7","8","9",
           "Mup","Mdown","Mleft","Mright","M1","M2","M3up","M3down",
           "Shift","Caps","Ctrl","Space","Esc","Enter","Windows","Backspace","Alt",
           "Question","Period","Exclamation","ParaL","ParaR","QuoteL","QuoteR",))
SYMB_CONV = {'ParaL':'(','ParaR':')','Question':'?','Exclamation':'!','Period':'.','QuoteL':"'",'QuoteR':"'",}
THRESHOLD=0.5
assert len(SYMBOLS) == 60, f"Invalid Symbol Amount {len(SYMBOLS)}"
SYMBOL_MAPS = {c:i for i,c in enumerate(SYMBOLS)}
INV_SYMBOLS = {i:c for c,i in SYMBOL_MAPS.items()}


