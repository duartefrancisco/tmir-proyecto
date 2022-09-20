from turtle import color
import cv2

face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_smile.xml")

def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0, 255), 3)
        roi_gray = gray[y: y + h, x : x + w]
        roi_color = frame[y: y + h, x : x + w]
        #cv2.imshow("ROI", roi_gray)
        #cv2.imshow("ROI", roi_color)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        for (sx, sy, sw, sh) in smiles:
          #  cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0,0, 255), 3)
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255, 0), 3)
          cv2.putText(frame, "Sonriendo", (x+sx, y+sy+sh), fontScale = 1, fontFace = cv2.FONT_HERSHEY_TRIPLEX, color= (255,255,255))

    return frame

video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_salida = detect(gray, frame)
    cv2.imshow("Salida", frame_salida)

    if(cv2.waitKey(1) & 0xFF == ord("q")):
        break

video_capture.release()
cv2.destroyAllWindows()