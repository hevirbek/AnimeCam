# Gerekli Kütüphaneler
import torch
import cv2
from PIL import Image
import numpy as np


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Kullanacağımız hazır model,  "celeba_distill", "paprika" veya "face_paint_512_v1"
# de denenebilir.
PRETRAINED = "face_paint_512_v2"

face2paint = torch.hub.load(
    "bryandlee/animegan2-pytorch:main", "face2paint", device=DEVICE)
model = torch.hub.load("bryandlee/animegan2-pytorch:main",
                       "generator", device=DEVICE, pretrained=PRETRAINED)

# Kameramızı açıyoruz
cam = cv2.VideoCapture(0)
while(True):
    _, frame = cam.read()

    # Kameradan gelen görüntüyü PIL'e dönüştürüyoruz
    frame = frame[:, :, [2, 1, 0]]
    frame = Image.fromarray(frame)

    # Görüntüyü modelimize gönderiyoruz
    frame = face2paint(model, frame)

    # Görüntüyü numpy array'a dönüştürüyoruz
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    # Görüntüyü ekrana çıkartıroyuz
    cv2.imshow('frame', frame)

    # q tuşuna basıldığında çıkış yapıyoruz
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Kamerayı kapatıyoruz
cam.release()

cv2.destroyAllWindows()
