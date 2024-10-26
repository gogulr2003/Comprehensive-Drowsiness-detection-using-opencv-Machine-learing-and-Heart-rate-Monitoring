import cv2
import dlib
from imutils import face_utils

# Load the pre-trained HOG face detector from dlib
detector = dlib.get_frontal_face_detector()

# Initialize the video s
vs = cv2.VideoCapture(0)  # You can change the argument to the video file path if needed


while True:
    # Read a frame from the video stream
    ret, frame = vs.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray, 1)

    # Loop over the detected faces
    for (i, face) in enumerate(faces):
        # Get the coordinates of the bounding box
        (x, y, w, h) = face_utils.rect_to_bb(face)

        # Draw the bounding box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the face number
        cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video stream and close all windows
vs.release()
cv2.destroyAllWindows()