import csv
import os
import cv2
import sys
import random


class Params:
    side = 64  # Desired side of the (square) image, in pixels
    margin = 0  # Desired margin (in pixels) around the bounding box of the image, must be >=0
    random_seed = 42
    prob = .36  #.18


# Load telemetry from the dataset, reading all .csv files in the given directory
input_dir = '/home/fanta/datasets/vehicles-detection/object-dataset'
input_fnames = [input_dir + '/labels.csv']
header = [False]
output_dir = '/home/fanta/datasets/vehicles-detection/object-dataset-select'
random.seed = Params.random_seed

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

records = []
for fname, has_header in zip(input_fnames, header):
    with open(fname) as csv_file:
        reader = csv.reader(csv_file)
        header = has_header
        for line in reader:
            if header:
                header = False
                continue
            fields = line[0].split(' ')
            assert 7 <= len(fields) <= 8
            if len(fields) == 7:
                filename, x0, y0, x1, y1, flag, kind = fields
            else:
                filename, x0, y0, x1, y1, flag, kind, _ = fields
            # Strip double-quotes from around the `kind` string
            kind = kind[1:-1]
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            # Sanity checks
            assert x1 > x0
            assert y1 > y0
            entry = {'filename': filename,
                     'x0': x0,
                     'y0': y0,
                     'x1': x1,
                     'y1': y1,
                     'flag': int(flag),
                     'kind': kind
                     }
            records.append(entry)

    print('Read', len(records), 'lines from input csv file', fname)


def expand(a0, b0, length):
    assert b0 > a0 and length >= b0 - a0 + 1
    if b0 - a0 + 1 == length:
        return a0, b0
    delta = (length - (b0 - a0 + 1)) // 2
    b0 += delta
    a0 -= length - (b0 - a0 + 1)
    assert b0 - a0 + 1 == length and b0 > a0
    return a0, b0


image_shape = None
counter = 0
processed_counter = 0
for entry in records:
    x0, y0, x1, y1, filename, flag, kind = entry['x0'], \
                                           entry['y0'], \
                                           entry['x1'], \
                                           entry['y1'], \
                                           entry['filename'], \
                                           entry['flag'], \
                                           entry['kind']
    # I only want cars without occlusion
    if kind != 'car' or flag == 1:
        continue
    if random.random() > Params.prob:
        continue
    processed_counter += 1
    if processed_counter % 20 == 0:
        sys.stdout.write("\rProcessing entry# {0:>6}".format(processed_counter))
        sys.stdout.flush()

    image = cv2.imread(input_dir + '/' + filename)
    assert image is not None
    if image_shape is None:
        image_shape = image.shape
    else:
        assert image_shape == image.shape
    assert y1 < image_shape[0]
    assert x1 < image_shape[1]
    # Make the image square, if not already square, by adding pixels to it as necessary
    if y1 - y0 < x1 - x0:
        y0, y1 = expand(y0, y1, x1 - x0 + 1)
    elif x1 - x0 < y1 - y0:
        x0, x1 = expand(x0, x1, y1 - y0 + 1)
    assert y1 - y0 == x1 - x0
    margin = round(Params.margin * (x1 - x0 + 1))
    assert margin >= 0
    x0 -= margin
    x1 += margin
    y0 -= margin
    y1 += margin
    if x1 > image.shape[1] or x0 < 0 or y1 > image.shape[0] or y0 < 0:
        continue
    if x1 - x0 + 1 < Params.side:
        continue
    snippet = image[y0:y1 + 1, x0:x1 + 1, :]
    if x1 - x0 + 1 > Params.side:
        snippet = cv2.resize(snippet, (Params.side, Params.side), interpolation=cv2.INTER_AREA)

    out_fname = 'object-dataset-{:05d}.png'.format(counter)
    cv2.imwrite(output_dir + '/' + out_fname, snippet)

    counter += 1

print('\nWrote', counter, 'images into ', output_dir)
