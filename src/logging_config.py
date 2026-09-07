import logging
import os
import sys
 
_CONFIGURED = False
 
 
def _configure_root_logger() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
 
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
 
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
 
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)

    if level > logging.DEBUG:
        for noisy_logger_name in ("paddleocr", "ppocr", "PIL", "urllib3", "httpx"):
            logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)
 
    _CONFIGURED = True
 
 
def get_logger(name: str) -> logging.Logger:
    _configure_root_logger()
    return logging.getLogger(name)