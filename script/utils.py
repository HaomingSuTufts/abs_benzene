import logging
import time
import functools
import os
import json


def setup_logging(output_dir: str, task_name: str) -> None:
    """Set up logging to save to another output folder."""
    log_path = os.path.join(output_dir, f"{task_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def monitor_performance(func):
    """a decorator to monitor the performance of a function"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logging.info(f"{func.__name__} completed in {elapsed_time:.2f}s")
            return result
        except Exception as e:
            logging.error(f"{func.__name__} failed: {str(e)}")
            raise
    return wrapper
