import logging
import os
import sys

# Create a package-wide logger
logger = logging.getLogger("gpplus")
logger.setLevel(logging.CRITICAL)  # Default: Disable logging unless configured by the user

# Prevent log duplication if a handler already exists
if not logger.handlers:
    handler = logging.NullHandler()  # Default: No output
    logger.addHandler(handler)


def configure_logger(level=logging.INFO, log_to_file=None):
    """
    Configures the package-wide logger.

    Parameters:
    - level (int): Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING)
    - log_to_file (str | None): Optional file path; when set, logs to file and stdout
    """
    # Remove existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_to_file:
        log_dir = os.path.dirname(os.path.abspath(log_to_file))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_to_file))

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)

    outputs = ["console"]
    if log_to_file:
        outputs.append(f"file:{log_to_file}")
    logger.info(
        f"Logger configured: level={logging.getLevelName(level)}, output={'+'.join(outputs)}"
    )


# Ensure that if the logger has never been configured, we set it up with a default format
if not logger.hasHandlers():
    configure_logger(logging.WARNING)
