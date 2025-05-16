import numpy as np
import cv2
from PIL import Image
from time import time

def processImg(img):
    img_tensor = np.array(img).astype(np.float32)  # Convert image to float32
    img_tensor = np.expand_dims(img_tensor, axis=0)  # Add batch dimension
    return img_tensor

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path="C:\\Users\\Niranjana\\Downloads\\mobilenetv2_BSD (1).tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Use a video file instead of a camera
video_path = "C:\\Users\\Niranjana\\Downloads\\WhatsApp Video 2024-10-08 at 12.02.58 PM.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Couldn't open video file")
    exit()

font = cv2.FONT_HERSHEY_SIMPLEX

def getPredictionFromRetAndFrame(ret, frame):
    if not ret:
        print("Can't receive frame from video, skipping...")
        return
    
    timer = time()

    LINE_THICKNESS = 4  # Adjust thickness for larger resolution
    LINE_COLOUR = (0, 0, 255)
    
    # Draw prediction box lines on the frame (adjusted for 1920x1080)
    cv2.line(frame, (0, 400), (480, 500), LINE_COLOUR, thickness=LINE_THICKNESS)
    # cv2.line(frame, (400, 400), (800, 400), LINE_COLOUR, thickness=LINE_THICKNESS)
    cv2.line(frame, (1000, 480), (-100, 200), LINE_COLOUR, thickness=LINE_THICKNESS)
    
    # Crop and resize the frame for the model input
    if frame.shape[0] == 480 and frame.shape[1] ==848:  # Ensure frame is large enough
        cropped = frame[0:480, 0:848]
        resized = cv2.resize(cropped, (160, 160))

        # Preprocess image and predict
        interpreter.set_tensor(input_details[0]['index'], processImg(resized))
        interpreter.invoke()

        # Convert raw to confidence level using sigmoid
        pred_raw = interpreter.get_tensor(output_details[0]['index'])
        pred_sig = sigmoid(pred_raw)
        pred = np.where(pred_sig < 0.5, 0, 1)
        timer = time() - timer

        # Display the prediction result
        readable_val = winName if pred[0][0] == 0 else ""
        print(f"{readable_val} - Prediction took {round(timer, 3)}s")
        print("----------------------\n")

    else:
        print("Frame size too small for processing.")




while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of the video

    getPredictionFromRetAndFrame(ret, frame)

    # Display the frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()