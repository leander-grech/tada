"""Logging related utilities"""

import json
import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

DATEFMT_STR = "%Y-%m-%dT%H:%M:%S%z"


def get_logging_fmt_str() -> str:
    return "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON"""

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": self.formatTime(record, datefmt=DATEFMT_STR),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


def setup_logging():
    logging.getLogger("httpx").setLevel(logging.ERROR)

    root_logger = logging.getLogger()

    # Remove all existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if os.getenv("USE_NONSTRUCTURED_LOGGING"):
        logging.basicConfig(
            level=logging.DEBUG,
            format=get_logging_fmt_str(),
            datefmt=DATEFMT_STR,
        )
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
