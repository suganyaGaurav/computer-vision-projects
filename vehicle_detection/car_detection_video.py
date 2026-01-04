"""
Car Detection in Video using Haar Cascades
"""

import cv2

car_cascade = cv2.CascadeClassifier("cascades/HaarCascade_cars.xml")

def detect_cars(frame):
    cars = car_cascade.detectMultiScale(frame, scaleFactor=1.15, minNeighbors=4)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

def run_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output = detect_cars(frame)
        cv2.imshow("Car Detection", output)

        if cv2.waitKey(2) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_video("sample_data/moving-cars_I.mp4")
