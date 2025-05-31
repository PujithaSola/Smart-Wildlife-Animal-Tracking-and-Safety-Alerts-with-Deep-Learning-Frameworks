import torch
from torchvision import models
import cv2
import numpy as np
from torchvision import transforms
import os
import platform

# Define the path to your saved model or weights
model_path = 'C:\\projects\\epics500\\trained_model-resnet.pth'  # Replace with your model file path

# Class names (update with your dataset classes)
class_names = ['Buffalo', 'Cheetah', 'Elephant', 'Lion', 'Tiger']

# Wild animals to trigger a beep sound
wild_animals = ['Cheetah', 'Elephant', 'Lion', 'Tiger']

# Function to play a beep sound
def play_beep():
    system = platform.system()
    if system == "Windows":
        import winsound
        winsound.Beep(1000, 500)  # Frequency = 1000 Hz, Duration = 500 ms
    elif system == "Darwin" or system == "Linux":
        os.system('echo -e "\a"')  # Unix-like systems terminal beep

# Function to load the model
def load_model(model_path):
    try:
        # Attempt to load the full model
        model = torch.load(model_path)
        model.eval()
        print("Loaded the entire model successfully.")
        return model
    except AttributeError:
        # If loading full model fails, assume it is a state dictionary
        print("Loaded state_dict; reconstructing the model architecture...")
        model = models.resnet18(pretrained=False)  # Replace with the trained architecture
        num_classes = len(class_names)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

# Load the model
model = load_model(model_path)

# Define image preprocessing transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # ResNet requires 224x224 input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Open the laptop camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Unable to access the laptop camera. Ensure it's not being used by another application.")

print("Press 'q' to quit the camera feed.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Preprocess the frame for the model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(frame_rgb).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds[0].item()]

    # Display the prediction on the video feed
    cv2.putText(frame, f"Pred: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Camera Feed', frame)

    # Play a beep sound if a wild animal is detected
    if predicted_class in wild_animals:
        print(f"Warning! Wild animal detected: {predicted_class}")
        play_beep()

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
