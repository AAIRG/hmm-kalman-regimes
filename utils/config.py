import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = "app_config.yaml") -> Dict[str, Any]:
    """
    Loads the general application and model configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration YAML file.
        
    Returns:
        A dictionary containing the configuration parameters.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Configuration file {config_path} not found. Returning empty config.")
        return {}
        
    try:
        with open(path, 'rt') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config if config else {}
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        return {}


def save_config(config: Dict[str, Any], config_path: str = "app_config.yaml") -> None:
    """
    Saves the application configuration to a YAML file.
    
    Args:
        config: The configuration dictionary to save.
        config_path: Path to the configuration YAML file.
    """
    path = Path(config_path)
    try:
        with open(path, 'wt') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Successfully saved configuration to {config_path}")
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {e}")
