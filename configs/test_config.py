import yaml
from types import SimpleNamespace


with open("configs\default_mappo.yaml", "r") as f:
    cfg_dict = yaml.safe_load(f)

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

cfg = dict_to_namespace(cfg_dict)

# print(cfg)

# cfg = SimpleNamespace(**cfg)

assert cfg.ppo.share_parameters_policy == False
print(cfg.ppo.clip_epsilon)