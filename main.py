from PIL import Image
import cv2
from mtcnn import mtcnn
import torch
from torch import nn
from torchvision import models
from pathlib import Path


class Linear(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.ln1 = nn.Linear(n_classes, 2)

    def forward(self, x):
        x = self.ln1(x)
        # x = self.ln6(x)
        # преобразование в одномерный вектор
        return x


my_lin_net = Linear(512)
my_net = models.resnet18()
my_net.fc = my_lin_net

my_net = torch.load(Path("net/net.pth"), map_location=torch.device("cpu"))
my_net.eval()

cap = cv2.VideoCapture(0)

mt_nn = mtcnn.MTCNN(Path("mtcnn"))

while True:
    ret, img = cap.read()
    cv2.imshow("camera", img)

    my_img = Image.fromarray(img)

    bounds = mt_nn.detect_faces(my_img)
    if len(bounds[0]) == 0:
        continue
    x, y, w, h, p = bounds[0][0]
    cv2.imshow("camera", cv2.rectangle(img, (int(x), int(y)), (int(w), int(h)), (0, 255, 0)))
    img_s = list()
    img_s.append(img)
    # print(my_net(torch.Tensor(img_s)))
    if cv2.waitKey(10) == 27:
        cv2.destroyAllWindows()
        break
cap.release()
