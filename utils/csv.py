import csv
import os


def create_csv(csv_path, header):
    if not os.path.exists(csv_path):
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)


def append_csv(csv_path, row):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)