import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import platform
import subprocess

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red
MAX_RESULTS = 5
SHOW_RESULTS = True


def play_sound(file,volume,pitch,position):
    system = platform.system()
    if system == "Windows":
        process = subprocess.Popen(
            f"wsl export LD_LIBRARY_PATH=./SFML-3.0.0/lib && ./audio "
            f"{file} {volume} {pitch} {position[0]} {position[1]} {position[2]}", shell=False)
    elif system == "Linux":
        process = subprocess.Popen(f"export LD_LIBRARY_PATH=./SFML-3.0.0/lib && ./audio "
                                   f"{file} {volume} {pitch} {position[0]} {position[1]} {position[2]}", shell=False)
    else:
        return -1
    return process


def visualize(image,detection_result) -> np.ndarray:
    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    return image


def play_sounds(detection_result):
    global processes
    for p in processes:
        p.terminate()
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        sizex = bbox.width / res[0]
        sizey = bbox.height / res[1]
        location_x = bbox.origin_x / res[0]
        location_y = bbox.origin_x / res[0]
        size_avg = (sizex + sizey) / 2
        process = play_sound("BeepSFX.mp3", size_avg * 2, 1.0, (location_x, location_y, 0.0))
        processes.append(process)

model_path = "./efficientdet_lite0.tflite"
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    max_results=MAX_RESULTS,
    running_mode=VisionRunningMode.VIDEO)
detector = ObjectDetector.create_from_options(options)


def main():
    global seen, res, processes
    # res = (640,480)
    res = (1280, 960)
    curr_frame = 0
    ptime = 0
    processes = []
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    seen = False
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    try:
        while True:
            ret, frame = cap.read()
            ctime = time.time()
            fps = 1 / (ctime - ptime)
            ptime = ctime
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = detector.detect_for_video(mp_image, round(curr_frame))
            image_copy = np.copy(mp_image.numpy_view())
            if SHOW_RESULTS:
                annotated_image = visualize(image_copy, result)
            else:
                annotated_image = image_copy
            play_sounds(result)
            image = annotated_image #cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (32, 32, 255), 3)
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame', image)
            cv2.waitKey(1)
            curr_frame += 1
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()


if __name__ == "__main__":
    main()
