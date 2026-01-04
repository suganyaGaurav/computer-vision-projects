"""
License Plate Detection using Haar Cascades
"""

import cv2
import matplotlib.pyplot as plt

plate_cascade = cv2.CascadeClassifier(
    "cascades/haarcascade_russian_plate_number.xml"
)

def detect_plate(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.2)
    print(f"Number of plates detected: {len(plates)}")

    for (x, y, w, h) in plates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    detect_plate("sample_data/car_plate.jpg")
