import os
from typing import Any, Dict

import yaml


class ConfigLoader:
    def __init__(self, configs_dir: str = None):
        """
        Initialize ConfigLoader with optional custom configs directory.
        If not provided, will use the directory of the current file.
        """
        self.configs_dir = configs_dir if configs_dir else os.path.dirname(__file__)

    def load(self, config_name: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_name: Name of the config file (without .yaml/.yml extension)

        Returns:
            Dictionary with configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If there's an error parsing YAML
        """
        if not config_name.lower().endswith(('.yaml', '.yml')):
            config_name += '.yaml'

        config_path = os.path.join(self.configs_dir, config_name)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found")

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_configs_dir(self) -> str:
        """Get the current configs directory"""
        return self.configs_dir

    def set_configs_dir(self, new_dir: str):
        """Set a new configs directory"""
        self.configs_dir = new_dir
