import atexit
import csv
from datetime import datetime
import logging
from logging.handlers import MemoryHandler
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
    
def make_logger(output_dir: str, name: str=None, console_level=logging.INFO, file_level=logging.INFO, csv_level=logging.INFO, delay=True) -> logging.Logger:
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
    if delay:
        file_handler = MemoryHandler(capacity=10000, target=file_handler, flushLevel=logging.CRITICAL)
    file_handler.setLevel(file_level)
    file_handler.addFilter(NotListFilter())
    csv_handler = CSVHandler(os.path.join(output_dir, name) + ".csv")
    if delay:
        csv_handler = MemoryHandler(capacity=10000, target=csv_handler, flushLevel=logging.CRITICAL)
    csv_handler.addFilter(ListFilter())
    csv_handler.setLevel(csv_level)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(csv_handler)
    
    atexit.register(logging.shutdown)
    
    return logger

def flush_delayed_logger(logger: logging.Logger):
    logger.handlers[1].flush() # file_handler
    logger.handlers[2].flush() # csv_handler

def log_memory_usage(logger: logging.Logger):
    process = psutil.Process(os.getpid())
    logger.info(f"Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")
    
def is_running_under_slurm() -> bool:
    slurm_vars = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NODELIST",
        "SLURM_SUBMIT_DIR",
        "SLURM_CLUSTER_NAME",
    ]
    return any(var in os.environ for var in slurm_vars)