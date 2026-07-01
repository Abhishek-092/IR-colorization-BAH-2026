import os
import logging
import json
from datetime import datetime

class JsonFormatter(logging.Formatter):
    """Formats log records as structured JSON."""
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage()
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

def setup_varna_logger(output_dir):
    """
    Sets up a dual-handler logger (stdout + local JSON log file).
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "varna_execution.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Console Handler (Human-readable)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File Handler (Structured JSON)
    file_handler = logging.FileHandler(log_file)
    json_formatter = JsonFormatter()
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logger initialized. File target: {log_file}")
    return logger
