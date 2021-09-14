import cv2
from posenet.process_images import process_images
cap = cv2.VideoCapture("C:/Users/sriva/Documents/Allnode/human_video/Boy.mp4")

if (cap.isOpened()== False):
    print('Eror')

while(cap.isOpened()):
    ret, frame= cap.read()
    if ret==True:
        process_images(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25)== ord('q'):
            break
    else:
        break

cap.release()
cap.destroyAllWindows() 
