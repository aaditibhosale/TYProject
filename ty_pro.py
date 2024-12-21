

# chatgpt



import sys 
# it provides access to some variables used or maintained by the python
import cv2 
# real time computer vision.
import time
# it used to measure the time taken for certain operation
import numpy as np
# it support lager multi-dimensional array
from scipy.spatial import distance as dist
# cal. distance between points
from threading import Thread
# run multiple 

import pygame 
import queue
import mediapipe as mp

FACE_DOWNSAMPLE_RATIO = 1.5
RESIZE_HEIGHT = 460

thresh = 0.27
mouth_thresh = 0.5 
#sound_path = "alarm.wav"
pygame.mixer.init()
pygame.mixer.music.load('C:/Users/Aaditi/alarm.wav')
#pygame.mixer.music.play()

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

leftEyeIndex = [362, 385, 387, 263, 373, 380]
rightEyeIndex = [33, 160, 158, 133, 153, 144]
mouthIndex = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 375]  # Landmarks for mouth

blinkCount = 0
openMouthCount = 0
drowsy = 0
state = 0
mouthStatus = 0

# Define global variables to track yawning state
yawnCount = 0
yawning = 0
yawnState = 0
yawnDuration = 2.0  # Yawn duration threshold (in seconds)

blinkTime = 0.15  # 150ms
drowsyTime = 1.5  # 1500ms
ALARM_ON = False
GAMMA = 1.5
threadStatusQ = queue.Queue()

invGamma = 1.0 / GAMMA
table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(0, 256)]).astype("uint8")

def gamma_correction(image):
    return cv2.LUT(image, table)

def histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)  

def soundAlert(audio_file_path, threadStatusQ):
    pygame.mixer.music.load(audio_file_path)
    pygame.mixer.music.play()

    while True:
        if not threadStatusQ.empty():
            FINISHED = threadStatusQ.get()
            if FINISHED:
                pygame.mixer.music.stop()
                break

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[3], mouth[5])
    D = dist.euclidean(mouth[0], mouth[4])
    mar = (A + B + C) / (3.0 * D)
    return mar

def checkEyeStatus(landmarks, frame):
    mask = np.zeros(frame.shape[:2], dtype=np.float32)
    
    hullLeftEye = [(landmarks[i][0], landmarks[i][1]) for i in leftEyeIndex]
    cv2.fillConvexPoly(mask, np.int32(hullLeftEye), 255)

    hullRightEye = [(landmarks[i][0], landmarks[i][1]) for i in rightEyeIndex]
    cv2.fillConvexPoly(mask, np.int32(hullRightEye), 255)

    leftEAR = eye_aspect_ratio(hullLeftEye)
    rightEAR = eye_aspect_ratio(hullRightEye)
    ear = (leftEAR + rightEAR) / 2.0

    eyeStatus = 1  # 1 -> Open, 0 -> closed
    if ear < thresh:
        eyeStatus = 0

    return eyeStatus


def checkMouthStatus(landmarks):
    mouth = [(landmarks[i][0], landmarks[i][1]) for i in mouthIndex]
    mar = mouth_aspect_ratio(mouth)
    
    mouthStatus = 0  # 0 -> Closed, 1 -> Open
    if mar > mouth_thresh:
        mouthStatus = 1
    
    return mouthStatus


def checkBlinkStatus(eyeStatus):
    global state, blinkCount, drowsy
    if state >= 0 and state <= falseBlinkLimit:
        if eyeStatus:
            state = 0
        else:
            state += 1
    elif state >= falseBlinkLimit and state < drowsyLimit:
        if eyeStatus:
            blinkCount += 1
            state = 0
        else:
            state += 1
    else:
        if eyeStatus:
            state = 0
            drowsy = 1
            blinkCount += 1
        else:
            drowsy = 1


def checkYawnStatus(mouthStatus):
    global openMouthCount, yawnCount
    if mouthStatus == 1:
        openMouthCount += 1
        if openMouthCount >= yawnThreshold:
            yawnCount += 1
            openMouthCount = 0  # Reset after detecting a yawn
    else:
        openMouthCount = 0  # Reset if mouth is closed

capture = cv2.VideoCapture(0)

# Warm up camera
for i in range(10):
    ret, frame = capture.read()

# Simulate opening a camera
def open_camera():
    # Record the time when the camera is opened
    camera_open_time = time.time()
    print(f"Camera opened at {camera_open_time} seconds since the epoch")

    # Simulate some delay for the sake of example
    time.sleep(2)

    # Record the time when the camera is closed
    camera_close_time = time.time()
    print(f"Camera closed at {camera_close_time} seconds since the epoch")

    # Calculate the duration the camera was open
    duration_open = camera_close_time - camera_open_time
    print(f"Camera was open for {duration_open} seconds")


# Example usage
open_camera()

# Variables for calibration
totalTime = 0.0
validFrames = 0
dummyFrames = 100



print("Calibration in Progress!")
with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
    while validFrames < dummyFrames:
        validFrames += 1
        t = time.time()
        ret, frame = capture.read()
        height, width = frame.shape[:2]
        IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
        frame = cv2.resize(frame, None, fx=1 / IMAGE_RESIZE, fy=1 / IMAGE_RESIZE, interpolation=cv2.INTER_LINEAR)
        
        adjusted = histogram_equalization(frame)

        results = face_mesh.process(cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB))
        timeLandmarks = time.time() - t

        if not results.multi_face_landmarks:
            validFrames -= 1
            cv2.putText(frame, "Unable to detect face, Please check proper lighting", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "or decrease FACE_DOWNSAMPLE_RATIO", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow("Blink Detection Demo", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                sys.exit()
        else:
            totalTime += timeLandmarks

print("Calibration Complete!")

spf = totalTime / dummyFrames
print(f"Current SPF (seconds per frame) is {spf * 1000:.2f} ms")

drowsyLimit = drowsyTime / spf
falseBlinkLimit = blinkTime / spf
yawnThreshold = 15  # Number of frames the mouth needs to be open to count as a yawn
print(f"drowsy limit: {drowsyLimit}, false blink limit: {falseBlinkLimit}")




if __name__ == "__main__":
    vid_writer = cv2.VideoWriter('output-low-light-2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (frame.shape[1], frame.shape[0]))

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        while True:
            try:
                t = time.time()
                ret, frame = capture.read()
                
                if not ret:
                    print("Failed to capture frame from camera.")
                    break

                # Frame resizing and adjustment
                height, width = frame.shape[:2]
                IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
                frame = cv2.resize(frame, None, fx=1 / IMAGE_RESIZE, fy=1 / IMAGE_RESIZE, interpolation=cv2.INTER_LINEAR)
                
                adjusted = histogram_equalization(frame)
                results = face_mesh.process(cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB))

                # Handle no face detection
                if not results.multi_face_landmarks:
                    cv2.putText(frame, "No face detected. Check lighting.", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow("Drowsiness Detection", frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
                        break
                    continue

                # Process facial landmarks for eye and mouth
                landmarks = [(lm.x * width, lm.y * height) for lm in results.multi_face_landmarks[0].landmark]
                eyeStatus = checkEyeStatus(landmarks, frame)
                checkBlinkStatus(eyeStatus)
                mouthStatus = checkMouthStatus(landmarks)
                checkYawnStatus(mouthStatus)

                # Display indicators and handle alarm
                if drowsy:
                    cv2.putText(frame, "DROWSINESS ALERT!", (70, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    if not ALARM_ON:
                        ALARM_ON = True
                        threadStatusQ.put(not ALARM_ON)
                        thread = Thread(target=soundAlert, args=('C:/Users/Aaditi/alarm.wav', threadStatusQ,))
                        thread.setDaemon(True)
                        thread.start()
                else:
                    cv2.putText(frame, f"Blinks: {blinkCount}", (10, 80), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Open Mouths: {openMouthCount}", (10, 110), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Yawns: {yawnCount}", (10, 140), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
                    ALARM_ON = False

                cv2.imshow("Drowsiness Detection", frame)
                vid_writer.write(frame)

                if cv2.waitKey(1) == ord('q'):  # Exit on 'q'
                    threadStatusQ.put(True)
                    break

            except Exception as e:
                print(f"Error: {e}")
                break

    vid_writer.release()
    capture.release()
    cv2.destroyAllWindows()






















