"""Configuration for one bounded adaptive XGBoost study."""
from dataclasses import dataclass, field, fields
from copy import deepcopy
import json
import math
from pathlib import Path


def default_search_space():
    return {
        "max_depth": {"type":"int", "low":1, "high":4},
        "n_estimators": {"type":"categorical", "choices":[50,100,200,500,1000]},
        "learning_rate": {"type":"float", "low":0.01, "high":0.2, "log":True},
        "min_child_weight": {"type":"float", "low":1.0, "high":20.0, "log":True},
        "reg_lambda": {"type":"float", "low":0.1, "high":100.0, "log":True},
        "subsample": {"type":"float", "low":0.6, "high":1.0, "log":False},
        "colsample_bytree": {"type":"float", "low":0.5, "high":1.0, "log":False},
    }


@dataclass(frozen=True, slots=True)
class XGBoostOptunaConfig:
    experiment_id: str = "e5b_optuna_ext02"
    base_config_path: Path = Path("configs/e5b_xgboost.json")
    output_dir: Path = Path("outputs/results/e5b_optuna_ext02_run_01")
    n_trials: int = 30
    sampler_seed: int = 42
    n_startup_trials: int = 10
    search_space: dict = field(default_factory=default_search_space)

    @staticmethod
    def project_root():
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def reference_parameters():
        common = dict(learning_rate=0.05,min_child_weight=5.0,reg_lambda=10.0,
                      subsample=1.0,colsample_bytree=1.0)
        return [dict(common,max_depth=2,n_estimators=100),
                dict(common,max_depth=4,n_estimators=1000)]

    def __post_init__(self):
        if not isinstance(self.experiment_id,str) or not self.experiment_id or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in self.experiment_id):
            raise ValueError("experiment_id must be a safe non-empty filename component.")
        for name,minimum in (("n_trials",2),("sampler_seed",0),("n_startup_trials",1)):
            value=getattr(self,name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if self.sampler_seed > 2**32-1:
            raise ValueError("sampler_seed is out of range.")
        for name in ("base_config_path","output_dir"):
            value=getattr(self,name)
            if not isinstance(value,(str,Path)) or not str(value).strip():
                raise ValueError(f"Invalid {name}.")
            object.__setattr__(self,name,(self.project_root()/Path(value).expanduser()).resolve())
        space=deepcopy(self.search_space)
        expected=default_search_space()
        if not isinstance(space,dict) or set(space)!=set(expected):
            raise ValueError("Search space must contain exactly the seven supported parameters.")
        for name,spec in space.items():
            if not isinstance(spec,dict) or spec.get("type")!=expected[name]["type"]:
                raise ValueError(f"Invalid distribution for {name}.")
            if spec["type"]=="categorical":
                if set(spec)!={"type","choices"}:
                    raise ValueError("Invalid categorical distribution fields.")
                choices=spec["choices"]
                if not isinstance(choices,list) or not choices or any(type(v) is not int or v<=0 for v in choices) or len(set(choices))!=len(choices):
                    raise ValueError("n_estimators choices must be unique positive integers.")
            else:
                allowed={"type","low","high"} if spec["type"]=="int" else {"type","low","high","log"}
                if set(spec)!=allowed:
                    raise ValueError(f"Invalid distribution fields for {name}.")
                low,high=spec["low"],spec["high"]
                if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (low,high)) or not 0<low<=high:
                    raise ValueError(f"Invalid search bounds for {name}.")
                if spec["type"]=="int" and (type(low) is not int or type(high) is not int):
                    raise ValueError("max_depth bounds must be integers.")
                if spec["type"]=="float" and type(spec["log"]) is not bool:
                    raise ValueError("log must be boolean.")
                if name in ("learning_rate","subsample","colsample_bytree") and high>1:
                    raise ValueError(f"Upper bound of {name} must not exceed one.")
        for anchor in self.reference_parameters():
            for name,value in anchor.items():
                spec=space[name]
                if (value not in spec["choices"] if spec["type"]=="categorical" else not spec["low"]<=value<=spec["high"]):
                    raise ValueError(f"Search space excludes a queued reference value for {name}.")
        object.__setattr__(self,"search_space",space)

    @classmethod
    def from_json(cls,path,project_root=None):
        root=Path(project_root).resolve() if project_root is not None else cls.project_root()
        raw=json.loads((root/Path(path).expanduser()).read_text(encoding="utf-8"))
        if not isinstance(raw,dict):
            raise ValueError("Optuna config must be a JSON object.")
        unknown=set(raw)-{f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown Optuna settings: {sorted(unknown)}.")
        defaults={"base_config_path":"configs/e5b_xgboost.json",
                  "output_dir":"outputs/results/e5b_optuna_ext02_run_01"}
        for name,default in defaults.items():
            raw[name]=(root/Path(raw.get(name,default)).expanduser()).resolve()
        return cls(**raw)

    def to_dict(self):
        return {f.name:str(getattr(self,f.name)) if isinstance(getattr(self,f.name),Path)
                else deepcopy(getattr(self,f.name)) for f in fields(self)}
