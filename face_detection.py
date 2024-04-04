import cv2

# Load the cascade for face detection.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Specify the source of the video.
source = "/Users/useradmin/Desktop/opencv/Goku doing job of security guard during the party.mp4"
cap = cv2.VideoCapture(source)

# Check if the video source has been opened successfully.
if not cap.isOpened():
    print("Error opening video source")
    exit()

win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)  # Use WINDOW_NORMAL to allow resizing the window.

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale for face detection.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=11, minSize=(20, 20))

    # Draw rectangles around the faces.
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the frame with detected faces.
    cv2.imshow(win_name, frame)

    # Break the loop when 'ESC' is pressed.
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for 'ESC'
        break

# Release the video capture object and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
