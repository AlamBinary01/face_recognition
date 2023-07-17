import cv2 as cv

face_detect = cv.CascadeClassifier("/home/httpstealer/.local/lib/python3.10/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml")
video_cap = cv.VideoCapture(0)

while True:
    ret, video_data = video_cap.read()
    if not ret:
        break

    col = cv.cvtColor(video_data, cv.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE
    )
    
    for (x, y, w, h) in faces:
        cv.rectangle(video_data, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv.imshow("Video Live", video_data)
    
    if cv.waitKey(10) == ord("a"):
        break

video_cap.release()
cv.destroyAllWindows()


