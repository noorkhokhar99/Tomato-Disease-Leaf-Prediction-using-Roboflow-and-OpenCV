
import cv2
from roboflow import Roboflow

# Initialize Roboflow

rf = Roboflow(api_key="89oDCCc4u1bHAaLGbEJJ")
project = rf.workspace("absolute-foods-ownqh").project("tomato-disease-b518h")
model = project.version(3).model







# Open a connection to the webcam (use 0 for the default camera)
cap = cv2.VideoCapture("Tomato Diseases.mp4")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform prediction
    predictions = model.predict(frame, confidence=43, overlap=50).json()

    # Draw bounding boxes on the frame
    for prediction in predictions['predictions']:
        x, y, w, h = (
            int(prediction['x']),
            int(prediction['y']),
            int(prediction['width']),
            int(prediction['height'])
        )
        confidence = prediction['confidence']
        class_name = prediction['class']

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display class name and confidence
        text = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('Predictions', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
