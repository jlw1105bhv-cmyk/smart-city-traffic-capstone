import logging
import os


def setup_logging(log_file):

    os.makedirs(
        "logs",
        exist_ok=True
    )

    logger = logging.getLogger()

    if logger.handlers:
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8"
    )

    console_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)