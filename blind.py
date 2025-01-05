import cv2
import mediapipe as mp
import time
import math as math
import pygame

pygame.init()

class PoseTrackingDynamic:
    def __init__(self, mode=False, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.poseMp = mp.solutions.pose
        self.poses = self.poseMp.Pose(
            static_image_mode=mode,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.mpHolistic = mp.solutions.holistic

    def findFingers(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.poses.process(imgRGB)
        if self.results.pose_landmarks:
            # for poseLms in self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks, self.poseMp.POSE_CONNECTIONS)

        return frame

    def findPosition(self, frame, poseNo=0, draw=True):
        global seen, size_avg
        xList = []
        yList = []
        bbox = []
        self.lmsList = []
        if self.results.pose_landmarks:
            myPose = self.results.pose_landmarks
            for id, lm in enumerate(myPose.landmark):

                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmsList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            # print("Poses Keypoint")
            # print(bbox)
            if seen == False:
                pygame.mixer.music.play()
                seen = True
            if draw:
                cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                          (0, 255, 0), 2)
            sizex = (xmax-xmin)/res[0]
            sizey = (ymax-ymin)/res[1]
            size_avg = (sizex+sizey)/2
            print(size_avg)
        else:
            seen = False
            size_avg = None
            pygame.mixer.music.stop()

        return self.lmsList, bbox



def main():
    global seen, res, size_avg
    pygame.mixer.music.load("BeepSFX.mp3")
    # res = (640,480)
    res = (1280, 960)
    ctime = 0
    ptime = 0
    size_avg = None
    cap = cv2.VideoCapture(0)
    detector = PoseTrackingDynamic()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    seen = False
    pygame.mixer.music.set_volume(1)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()

        frame = detector.findFingers(frame)
        lmsList = detector.findPosition(frame)
        if len(lmsList) != 0:
            pass
        # print(lmsList[0])

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (32, 32, 255), 3)
        if size_avg:
            cv2.putText(frame, str(round(float(size_avg),5)), (100, 70), cv2.FONT_HERSHEY_PLAIN, 3, (32, 32, 255), 3)
            pygame.mixer.music.set_volume(round(float(size_avg),5)*2)
        else:
            cv2.putText(frame, "NaN", (100, 70), cv2.FONT_HERSHEY_PLAIN, 3, (32, 32, 255), 3)

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
