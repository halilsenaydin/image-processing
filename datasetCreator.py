import cv2
import os

# '\' | '/'
brace = os.sep

root = r'D:\Documents\Tutorial And Installer\Artificial Intelligence\Computer Vision\OpenCv\Algorithms\Haar Cascade'.replace(f'\\', brace)
detector=cv2.CascadeClassifier(f'{root}\\haarcascade_frontalface_default.xml'.replace(f'\\', brace))

userId=input('ID numarası giriniz: ')
userName=input('Kişi İsmi giriniz: ')

if not os.path.exists(f'.\\faces\\face-{userId}'.replace(f'\\', brace)):
    os.makedirs(f'.\\faces\\face-{userId}'.replace(f'\\', brace)) # Create Folder

webcam = cv2.VideoCapture(0)

i = 1
counter = len(os.listdir(f'.\\faces\\face-{userId}'.replace(f'\\', brace)))
print(counter)
while True:
    _, frame = webcam.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x+w,y+h), (255, 0, 255))

        path = f'.\\faces\\face-{userId}\\{userName}-{counter+i}.jpg'.replace(f'\\', brace)
        cv2.imwrite(path, gray[y:y+h, x:x+w])

        i = i + 1
        print(f'Kalan Poz: {100 - i}')

    cv2.imshow('Scanning Data...', frame)
    if (cv2.waitKey(10) & 0xFF == ord('x')) or i>100: # Collecting face data is sufficient:
        break

webcam.release()
cv2.destroyAllWindows()