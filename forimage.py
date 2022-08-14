# Gerekli Kütüphaneler
import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Kullanacağımız hazır model,  "celeba_distill", "paprika" veya "face_paint_512_v1"
# de denenebilir.
PRETRAINED = "face_paint_512_v2"  # "face_paint_512_v2"

face2paint = torch.hub.load(
    "bryandlee/animegan2-pytorch:main", "face2paint", device=DEVICE)
model = torch.hub.load("bryandlee/animegan2-pytorch:main",
                       "generator", device=DEVICE, pretrained=PRETRAINED)


# Fotoğrafı model ile çeviriyoruz ve çıktıyı kaydediyoruz
img = Image.open('test.jpg')
img = face2paint(model, img)
img.save('output.jpg')
