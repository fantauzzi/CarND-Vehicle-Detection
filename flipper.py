import glob
import cv2
from os.path import basename, splitext
import numpy as np

for fname in glob.glob('/home/fanta/datasets/vehicles-detection/non-vehicles-additional/*.png'):
    bname = basename(fname)
    stripped = splitext(bname)[0]
    out = stripped+'-flip.png'
    image = cv2.imread(fname)
    assert image is not None
    image = np.fliplr(image)
    cv2.imwrite(out, image)