import logging
import logging.config
import yaml
from pathlib import Path

def setup_logging(config_path: str = "config.yaml") -> None:
    """
    Sets up the logging configuration from a YAML file.
    
    Args:
        config_path: The path to the logging configuration file.
    """
    path = Path(config_path)
    if path.exists():
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
        logging.debug(f"Logging configured from {config_path}")
    else:
        logging.basicConfig(level=logging.INFO)
        logging.warning(f"Logging config file {config_path} not found. Falling back to basic configuration.")
