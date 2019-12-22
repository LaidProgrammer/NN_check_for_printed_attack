from PIL import Image
import cv2
import numpy as np
from mtcnn import mtcnn
import torch
from torch import nn
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision import models
from pathlib import Path


class Linear(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.ln1 = nn.Linear(n_classes, 2)

    def forward(self, x):
        x = self.ln1(x)
        return x


my_lin_net = Linear(512)
my_net = models.resnet18()
my_net.fc = my_lin_net

my_net.load_state_dict(torch.load(Path('./net/saved.pth'), map_location=torch.device("cpu")))
my_net.eval()

transform = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

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
    my_img = my_img.crop((int(x), int(y), int(x + w), int(y + h)))
    my_img = my_img.resize((196, 196))
    my_img = transform(my_img)
    test = my_net(my_img[None])
    test = nn.Softmax(dim=1)(test)
    test = test.detach().numpy()[0]
    test_out = np.argmax(test)
    color = (0, 0, 255)
    # print(test)
    if test_out == 0:
        color = (0, 255, 0)
    cv2.imshow("camera", cv2.rectangle(img, (int(x), int(y)), (int(w), int(h)), color))

    if cv2.waitKey(10) == 27:
        cv2.destroyAllWindows()
        break
cap.release()
