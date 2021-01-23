import cv2

# To capture video from webcam
# Might have to use 1 or 2 for parameter depending on if it's an integrated or usb webcam
cap = cv2.VideoCapture(0)

# Load Trained CascadeClassifier XML
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    # Read the frame
    ret, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(1) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()
#Delete any windows created
cv2.destroyallwindows()