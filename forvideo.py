# Gerekli Kütüphaneler
import torch
from PIL import Image
import numpy as np
import moviepy.editor as mpy


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Kullanacağımız hazır model,  "celeba_distill", "paprika" veya "face_paint_512_v1"
# de denenebilir.
PRETRAINED = "face_paint_512_v2"

face2paint = torch.hub.load(
    "bryandlee/animegan2-pytorch:main", "face2paint", device=DEVICE)
model = torch.hub.load("bryandlee/animegan2-pytorch:main",
                       "generator", device=DEVICE, pretrained=PRETRAINED)


# Kareyi model ile çeviriyoruz ve çıktıyı döndürüyoruz
def make_frame(t):
    img = Image.fromarray(t)
    img = face2paint(model, img)
    img = np.array(img)
    return img


# Videonun her karesi için bunu yapıp yeni video oluşturuyoruz
video = mpy.VideoFileClip('test.mp4')
video = video.fl_image(make_frame)
video.write_videofile('output.mp4', fps=30)
