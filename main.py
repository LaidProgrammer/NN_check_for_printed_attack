from PIL import Image
from mtcnn import mtcnn
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    cv2.imshow("camera", img)
    if cv2.waitKey(10) == 27: # Клавиша Esc
        break
cap.release()

cv2.destroyAllWindows()

# mtcnn_nn =

mrcnn_nn.detect_faces()
