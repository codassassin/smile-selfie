import cv2

video = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier("dataset/haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier("dataset/haarcascade_smile.xml")

while True:
    success, img = video.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayImg, 1.1, 4)
    count = 500
    keyPressed = cv2.waitKey(1)
    for x, y, w, h in faces:
        smiles = smileCascade.detectMultiScale(grayImg, 1.8, 15)
        for x, y, w, h in smiles:
            print("Image " + str(count) + "Saved")
            path = 'SavedImages\\' + str(count) + '.jpg'
            cv2.imwrite(path, img)
            count += 1
            if count >= 503:
                break

    cv2.imshow('live video', img)
    if keyPressed & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
