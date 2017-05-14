import csv
import os

# Load telemetry from the dataset, reading all .csv files in the given directory
input_fnames = ['/home/fanta/datasets/vehicles-detection/object-dataset/labels.csv']
header = [False]

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
            # Strip " from around the `kind` string
            kind = kind[1:-1]
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
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
