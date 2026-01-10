
import csv
import os

class CSVMetricLogger:
    def __init__(self, path: str, fieldnames: list[str], append: bool = False):
        self.path = path
        self.fieldnames = fieldnames
        self.append = append

        file_exists = os.path.exists(path)

        self.file = open(path, "a" if append else "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)

        if not append or not file_exists:
            self.writer.writeheader()
            self.file.flush()

    def log(self, row: dict):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()