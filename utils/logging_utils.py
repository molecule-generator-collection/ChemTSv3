import csv
from datetime import datetime
import logging
import os
import psutil

class CSVHandler(logging.Handler):
    def __init__(self, filename):
        super().__init__()
        self.baseFilename = filename # same name as FileHandler
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
    
def make_logger(output_dir: str, name: str=None, console_level=logging.INFO, file_level=logging.INFO, csv_level=logging.INFO) -> logging.Logger:
    if name is None:
        name = datetime.now().strftime("%m-%d_%H-%M")
    os.makedirs(output_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.addFilter(NotListFilter())
    file_handler = logging.FileHandler(os.path.join(output_dir, name) + ".log")
    file_handler.setLevel(file_level)
    file_handler.addFilter(NotListFilter())
    csv_handler = CSVHandler(os.path.join(output_dir, name) + ".csv")
    csv_handler.addFilter(ListFilter())
    csv_handler.setLevel(csv_level)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(csv_handler)
    
    return logger

def log_memory_usage(logger: logging.Logger):
    process = psutil.Process(os.getpid())
    logger.info(f"Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")