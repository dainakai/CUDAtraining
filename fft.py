import cv2
import numpy as np

imagepath = "./holograms/output_holo.bmp"
output = "./testfft2.png"

img = cv2.imread(imagepath,0)
# img = np.array(img)
img = np.fft.fft2(img)
img = np.fft.fftshift(img)
img = np.abs(img)
img = np.log(img + 1)
min = np.min(img)
max = np.max(img)

img = 255*(img - min)/(max-min)

cv2.imwrite(output, img)