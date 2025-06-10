import logging
import csv

class CSVHandler(logging.Handler):
    def __init__(self, filename):
        super().__init__()
        self.file = open(filename, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

    def emit(self, record):
        self.writer.writerow(record.msg) # should be tuple
        self.file.flush()

    def close(self):
        self.file.close()
        super().close()
        
class ListFilter(logging.Filter):
    def filter(self, record):
        return isinstance(record.msg, list)

class NotListFilter(logging.Filter):
    def filter(self, record):
        return not isinstance(record.msg, list)