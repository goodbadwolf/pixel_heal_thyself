"""Logger for the project."""

import logging
import sys
from typing import Any

from pht.utils import SingletonMeta


class Logger(metaclass=SingletonMeta):
    """Singleton logger class."""

    def __init__(self, log_level: str = "INFO") -> None:
        self._logger = logging.getLogger("pht")
        self.setup_logger(log_level)

    def setup_logger(self, log_level: str = "INFO") -> None:
        """Set up the logger."""
        log_level = log_level.upper()
        self._logger.setLevel(log_level)
        logging.getLogger("torch").setLevel(log_level)
        logging.captureWarnings(True)
        sys.stderr = self.StreamToLogger(self._logger, log_level)

    class StreamToLogger:
        """Class to redirect stderr to logger."""

        def __init__(
            self,
            logger: logging.Logger,
            log_level: str | int = "ERROR",
        ) -> None:
            self.logger = logger
            self.log_level = log_level
            if isinstance(log_level, str):
                self.log_level = getattr(logging, log_level)

        def write(self, buf: str) -> None:
            """Write buffer content to logger, splitting by lines."""
            for line in buf.rstrip().splitlines():
                self.logger.log(self.log_level, line.rstrip())

        def flush(self) -> None:
            """Implement flush operation for file-like interface."""

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        return getattr(self._logger, name)


logger = Logger()
