from mp_functions import *
from const import *

capture = cv.VideoCapture(0)

with mp_holistic.Holistic(min_tracking_confidence=0.5,min_detection_confidence=0.5) as holistic:
    while True:
        ret, frame = capture.read()



        results = mp_detection(frame,holistic)
        keypoints = extract(results)

        mp_draw(frame,results)

        cv.imshow('Feed', frame)


        if cv.waitKey(20)&0xFF == ord('x'):
            break