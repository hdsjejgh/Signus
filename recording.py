from logging import exception

from mp_functions import *
from const import *
rec = False
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import shutil

def create_folders(): #Creates all the folders
    for symbol in SYMBOLS: #makes a symbol for each folder
        try:
            #os.makedirs(os.path.join(DATA_DIR,symbol))
            for vid in range(60,90): #makes a directory for each video
                #each video will have 30 frame files in them
                if os.path.isdir(os.path.join(DATA_DIR, symbol,str(vid))):
                    os.rmdir(os.path.join(DATA_DIR, symbol,str(vid)))
                os.makedirs(os.path.join(DATA_DIR, symbol,str(vid)))
        except exception as e:
            print(f"Error Occurred when making folder for {symbol}. {e}")

def clear(symbol): #clear's a symbol's directory by removing and recreating it
    global rec
    shutil.rmtree(os.path.join(DATA_DIR,symbol))
    os.makedirs(os.path.join(DATA_DIR,symbol))
    for vid in range(VID_NUM):
        os.makedirs(os.path.join(DATA_DIR, symbol, str(vid)))
    rec = False

create_folders()

breakouter = False
currently_recording = "M3down"
if rec:
    capture = cv.VideoCapture(0)
    with mp_holistic.Holistic(min_tracking_confidence=0.5,min_detection_confidence=0.5) as holistic:
        for video in range(60,90):


            ret, frame = capture.read()

            results = mp_detection(frame, holistic)
            keypoints = extract(results)

            mp_draw(frame, results)
            nose = results.pose_landmarks.landmark[9]
            # print(nose.x)
            cv.putText(frame, f"Collecting for {currently_recording}. Video {video + 1}", org=(int(nose.x * 200), int(nose.y * 200)), fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=.5, color=(0, 0, 255))
            cv.imshow('Feed', frame)
            if cv.waitKey(2000)&0xFF == ord('x'):
                break

            for fr in range(FRAMES):
                ret, frame = capture.read()



                results = mp_detection(frame,holistic)
                keypoints = extract(results)

                mp_draw(frame,results)
                nose = results.pose_landmarks.landmark[9]
                #print(nose.x)
                cv.putText(frame,f"Collecting for {currently_recording}. Video {video+1}",org=(int(nose.x*200),int(nose.y*200)),fontFace=cv.FONT_HERSHEY_DUPLEX,fontScale=.5,color=(0,0,255))

                cv.imshow('Feed', frame)

                nppath = os.path.join(DATA_DIR,currently_recording,str(video),str(fr))
                np.save(nppath, keypoints)

                if cv.waitKey(20)&0xFF == ord('x'):
                    breakouter=True
                    break
            if breakouter:
                break
