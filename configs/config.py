import os
from typing import Dict, Iterable
from utils.yaml_model import YamlModel
from configs.llm_config import LLMConfig
from utils.constants import CONFIG_PATH, ROOT_PATH

class Config(YamlModel):
    llm: LLMConfig
    
    @classmethod
    def default(cls, reload: bool = False, **kwargs) -> "Config":
        default_config_paths = (
            ROOT_PATH / "configs" / "config.yaml",
            CONFIG_PATH / "config.yaml",
        )
        if reload or default_config_paths not in _CONFIG_CACHE:
            dicts = [dict(os.environ), *(Config.read_yaml(path) for path in default_config_paths), kwargs]
            final = merge_dict(dicts)
            _CONFIG_CACHE[default_config_paths] = Config(**final)
        return _CONFIG_CACHE[default_config_paths]
    
        
def merge_dict(dicts: Iterable[Dict]) -> Dict:
    """Merge multiple dicts into one, with the latter dict overwriting the former"""
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result

_CONFIG_CACHE = {}