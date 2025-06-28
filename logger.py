# in logger_setup.py
import logging
import sys

def setup_logger():
    # Create a logger object
    logger = logging.getLogger("CarScraper")
    
    # Avoid adding duplicate handlers if this function is called multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO) # Set the minimum level of messages to capture

    # Create a handler to print logs to the console (standard output)
    # This is what GitHub Actions will capture
    handler = logging.StreamHandler(sys.stdout)
    
    # Create a formatter to define the log message's structure
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(handler)
    
    return logger

# Create a single instance of the logger for other modules to import
log = setup_logger()