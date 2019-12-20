from PIL import Image
from mtcnn import mtcnn
import torch
import cv2
from pathlib import Path

cap = cv2.VideoCapture(0)

mt_nn = mtcnn.MTCNN(Path("mtcnn"))

# mynet = torch.load("./net.pth")

# mynet.eval()

while True:
    ret, img = cap.read()
    cv2.imshow("camera", img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    my_img = Image.fromarray(img)

    bounds = mt_nn.detect_faces(my_img)
    if len(bounds[0]) == 0:
        continue
    x, y, w, h, p = bounds[0][0]
    print(x, y, w, h, p)
    cv2.imshow("camera", cv2.rectangle(img, (int(x), int(y)), (int(w), int(h)), (255, 0, 0)))
    if cv2.waitKey(10) == 27:
        cv2.destroyAllWindows()
        break
cap.release()
