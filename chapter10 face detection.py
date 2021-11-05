import cv2

# facecascade = cv2.CascadeClassifier("resource/haarcascade_eye.xml") for eye
facecascade = cv2.CascadeClassifier("resource/haarcascade_frontalface_default.xml") 
# facecascade = cv2.CascadeClassifier("resource/haarcascade_russian_plate_number.xml") just for experiment
img = cv2.imread("resource/human.webp")
imggrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = facecascade.detectMultiScale(imggrey,1.1,4)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)


cv2.imshow("original",img)
cv2.waitKey(0)