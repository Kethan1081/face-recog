import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('F:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3,480)
cap.set(3,640)
rec = cv2.face.LBPHFaceRecognizer_create();
rec.read("F:/files/ProjectSCCL/trainingdata.yml")
id=0
font = cv2.FONT_HERSHEY_SIMPLEX
while 1:
    ret, img = cap.read()
    if ret is True:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        continue
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if id==2:
            id="Kethan"
        if id==1:
            id="Kethan"
        if id==3:
            id="Messi"
        if id==4:
            id="Neymar"
        if id==5:
            id='rahul'
        cv2.putText(img,str(id),(x,y),font,1,(0,255,0),2)    
    cv2.imshow('img',img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()