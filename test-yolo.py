from ultralytics import YOLO
import cv2
import math
import winsound

# Load the model
model = YOLO("C:\\projects\\epics100\\runs\\detect\\train2\\weights\\best.pt")
print("Model loaded successfully!")

# Object classes (update this based on your model)
classNames = ["Buffalo", "Cheetahs", "Deer", "Elephant", "Fox", "Jaguars", "Lion", "Panda", "Tiger", "Zebra"]  # Add more classes if needed

# List of target animals to beep for
target_animals = ["Buffalo", "Cheetahs", "Elephant", "Jaguars", "Lion", "Tiger",]  # Add any other animals you want to trigger a beep

# Function to play a beep sound
def beep():
    frequency = 1000  # Set Frequency To 1000 Hertz
    duration = 500  # Set Duration To 500 ms == 0.5 second
    winsound.Beep(frequency, duration)

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame!")
        break

    print("Frame read successfully!")

    # Perform detection
    results = model(img, show=True, conf=0.6)

    # Coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            # Class name
            cls = int(box.cls[0])
            animal_name = classNames[cls]
            print("Class index -->", cls)
            print("Class name -->", animal_name)

            # Check if the animal is in the target list
            if animal_name in target_animals:
                cv2.putText(img, f"{animal_name} (Target)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                beep()  # Beep for target animals
            else:
                cv2.putText(img, f"{animal_name} (Not Target)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Show the webcam feed with bounding boxes
    #cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
