import argparse
import cv2
import os
import matplotlib.pyplot as plt

# Description of this program.
desc = "Image cutter, to generate negative test samples"

# Create the argument parser.
parser = argparse.ArgumentParser(description=desc)

# Add arguments to the parser.
parser.add_argument("--input", required=True,
                    help="file name of the image")

parser.add_argument("--x0", required=True,
                    help="Upper left corner x coordinate")

parser.add_argument("--y0", required=True,
                    help="Upper left corner y coordinate")

parser.add_argument("--x1", required=True,
                    help="Lower right corner x coordinate")

parser.add_argument("--y1", required=True,
                    help="Lower right corner y coordinate")

# Parse the command-line arguments.
args = parser.parse_args()

# Get the arguments.
fname = args.input
roi = [[int(args.x0), int(args.y0)], [int(args.x1), int(args.y1)]]

image = cv2.imread(fname)

base_size = 64
step = base_size // 4
enlargements = range(1, 5)


def cutouts():
    def home(factor):
        """
        Returns the coordinates for the starting position of the detection window in the ROI (upper-left corner).
        """
        x0, y0 = roi[0]
        x1, y1 = x0 + base_size * factor - 1, y0 + base_size * factor - 1
        return x0, y0, x1, y1

    for enlargement in enlargements:
        x0, y0, x1, y1 = home(enlargement)
        if y1>roi[1][1]:
            return
        while True:
            # Sanity checks (sane paranoia)
            assert x1 - x0 + 1 == base_size * enlargement and y1 - y0 + 1 == base_size * enlargement
            assert x1 > x0 and y1 > y0
            assert x0 >= roi[0][0] and x0 <= roi[1][0]
            assert y0 >= roi[0][1] and y0 <= roi[1][1]
            assert x1 <= roi[1][0]
            assert y1 <= roi[1][1]

            yield x0, y0, x1, y1
            # Slide one step to the right
            x0 += step * enlargement
            x1 = x0 + base_size * enlargement - 1
            ''' If the window is now even partly out of the ROI to the right, slide one step down and return
            as left as possible '''
            if x1 > roi[1][0]:
                x0 = roi[0][0]
                x1 = x0 + base_size * enlargement - 1
                y0 += step * enlargement
                y1 = y0 + base_size * enlargement - 1
                ''' If the window is now even partly out of the ROI to the bottom, return it to the starting position,
                 at the top left of the roi '''
                if y1 > roi[1][1]:
                    break


no_ext, _ = os.path.splitext(fname)
progressive = 0
for cutout in cutouts():
    x0, y0, x1, y1 = cutout
    cut_image = image[y0:y1 + 1, x0:x1 + 1]
    assert cut_image.shape[0] % base_size == 0
    assert cut_image.shape[1] % base_size == 0
    if cut_image.shape[0] != base_size:
        cut_image = cv2.resize(cut_image, (base_size, base_size), interpolation=cv2.INTER_AREA)

    out_fname = no_ext + '-{:05d}'.format(progressive) + '.png'
    cv2.imwrite(out_fname, cut_image)
    progressive += 1

print('Images extracted:', progressive)
