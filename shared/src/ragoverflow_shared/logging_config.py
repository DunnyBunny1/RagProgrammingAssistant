import logging
import sys
from pathlib import Path


def setup_logging(module_path: str) -> logging.Logger:
    """
    Configure logging to write to both a log file and stdout.

    :param module_path: the `__file__` from the calling script
    :return: Configured logger instance
    """
    # Extract script name from module path
    script_name = Path(module_path).stem

    # Create handlers to write output to a log file & std-err
    file_handler = logging.FileHandler(
        filename=Path(__file__).parent.parent.parent.parent / "logs" / f"{script_name}.log",
        mode="w",
    )
    stream_handler = logging.StreamHandler(stream=sys.stdout)

    # Setup a logger w/ our handlers
    logging.basicConfig(
        handlers=[file_handler, stream_handler],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        force=True,
    )

    return logging.getLogger(__name__)
