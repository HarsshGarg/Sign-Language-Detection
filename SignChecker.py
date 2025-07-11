import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize camera
print("Initializing camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()
print("Camera initialized.")

# Initialize hand detector and classifier
print("Initializing hand detector and classifier...")
detector = HandDetector(maxHands=1)
try:
    classifier = Classifier(r"C:\Users\HP\Desktop\Model\keras_model.h5", r"C:\Users\HP\Desktop\Model\labels.txt")
    print("Classifier loaded successfully.")
except Exception as e:
    print("Error loading model or labels:", e)
    exit()

offset = 20
imgSize = 300

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
          "V", "W", "X", "Y", "Z"]

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        try:
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape
            print(f"Image cropped: {imgCropShape}")
        except Exception as e:
            print("Error cropping image:", e)
            continue

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            try:
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                print(f"Image resized (wCal): {imgResize.shape}")
            except cv2.error as e:
                print("Error resizing image:", e)
                continue
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            try:
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                print(f"Image resized (hCal): {imgResize.shape}")
            except cv2.error as e:
                print("Error resizing image:", e)
                continue
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        try:
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(f"Prediction: {prediction}, Index: {index}")
        except Exception as e:
            print("Error in classification:", e)
            continue

        cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0),
                      cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
