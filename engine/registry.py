from typing import Dict, Type

class Registry:
    """A lightweight class registry."""
    def __init__(self, name: str):
        self.name = name
        self.module_dict: Dict[str, Type] = {}

    def register(self, cls):
        """Register a class using decorator syntax."""
        name = cls.__name__
        if name in self.module_dict:
            raise KeyError(f"{name} already registered in {self.name}")
        self.module_dict[name] = cls
        return cls

    def get(self, name: str):
        """Retrieve a registered class by name."""
        if name not in self.module_dict:
            raise KeyError(f"{name} not found in {self.name}")
        return self.module_dict[name]

    def __contains__(self, name: str):
        return name in self.module_dict

    def __repr__(self):
        return f"Registry({self.name}, {list(self.module_dict.keys())})"



MODELS = Registry("models")
DATASETS = Registry("datasets")
LOSSES = Registry("losses")
HOOKS = Registry("hooks")  
