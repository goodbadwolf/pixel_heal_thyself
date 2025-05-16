import logging
import sys
import warnings

logger: logging.Logger = None


def setup_logger(log_level: str = "INFO"):
    log_level = log_level.upper()

    def redirect_warnings_to_logger(
        message, category, filename, lineno, file=None, line=None
    ):
        logger.warning(f"{category.__name__}: {message} ({filename}:{lineno})")

    class StreamToLogger:
        def __init__(self, logger, log_level: str | int = "ERROR"):
            self.logger = logger
            self.log_level = log_level
            if isinstance(log_level, str):
                self.log_level = getattr(logging, log_level)

        def write(self, buf):
            for line in buf.rstrip().splitlines():
                self.logger.log(self.log_level, line.rstrip())

        def flush(self):
            pass

    global logger
    logger = logging.getLogger("pht")
    logger.setLevel(log_level)

    # Capture torch logs
    logging.getLogger("torch").setLevel(log_level)

    # Redirect Python warnings to logger
    warnings.showwarning = redirect_warnings_to_logger

    # Optionally, redirect stderr (for C++/CUDA errors)
    sys.stderr = StreamToLogger(logger, log_level)


setup_logger()
