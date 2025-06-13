import cv2
import numpy as np
import os
import time
import pyttsx3

# Initialize text-to-speech
engine = pyttsx3.init()

# ====================
# Load YOLOv5
# ====================
print("[INFO] Loading YOLOv5 model...")
labelsPath = "model/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
with open(labelsPath, "r") as f:
    classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNetFromDarknet("model/yolov5.cfg", "model/yolov5.weights")
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# ====================
# Load Face Recognizer
# ====================
print("[INFO] Loading face recognizer...")
recognizer = cv2.face.LBPHFaceRecognizer_create()
if not os.path.exists('trainer/trainer.yml'):
    print("Trainer file not found!")
    exit()
recognizer.read('trainer/trainer.yml')

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
names = ['None', 'buvanes']  # Add more names as needed

# ====================
# Video Stream
# ====================
cam = cv2.VideoCapture(0)
#cam = cv2.VideoCapture('http://192.168.108.12:81/stream')
cam.set(3, 640)
cam.set(4, 480)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

frame_count = 0
flag = 1

print("[INFO] Starting video stream...")
while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = frame.shape[:2]

    # ====================
    # Face Detection & Recognition
    # ====================
    faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(int(minW), int(minH)))
    recognized_name = ""
    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        label_text = ""
        if confidence < 100:
            name = names[id]
            label_text = f"{name} ({round(100 - confidence)}%)"
            recognized_name = f"Hello {name}"
        else:
            name = "Unknown"
            label_text = f"{name} ({round(100 - confidence)}%)"
            recognized_name = "Unknown person detected"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label_text, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if recognized_name:
        print(recognized_name)
        engine.say(recognized_name)
        engine.runAndWait()

    # ====================
    # Object Detection (every 60 frames)
    # ====================
    if frame_count % 60 == 0:
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes = []
        confidences = []
        classIDs = []
        centers = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    centerX, centerY, width, height = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    centers.append((centerX, centerY))

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        texts = ["The environment has the following objects:"]

        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                color = colors[classIDs[i]]
                label = f"{classes[classIDs[i]]}: {confidences[i]:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                centerX, centerY = centers[i]
                W_pos = "left" if centerX < W/3 else "center" if centerX < 2*W/3 else "right"
                H_pos = "top" if centerY < H/3 else "middle" if centerY < 2*H/3 else "bottom"
                texts.append(f"{H_pos} {W_pos} - {LABELS[classIDs[i]]}")

            description = ', '.join(texts)
            print(description)
            engine.say(description)
            engine.runAndWait()

    # ====================
    # Display the frame
    # ====================
    cv2.imshow("Face and Object Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break

# ====================
# Cleanup
# ====================
print("\n[INFO] Exiting...")
cam.release()
cv2.destroyAllWindows()
