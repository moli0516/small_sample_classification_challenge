"""Settings for one fold-local feature-selection extension, not a new tuning search."""
from dataclasses import dataclass,fields
import json
from pathlib import Path


@dataclass(frozen=True,slots=True)
class XGBoostFeatureSelectionConfig:
    experiment_id:str="e5b_feature_selection_ext03"
    source_study_dir:Path=Path("outputs/results/e5b_optuna_ext02_run_01")
    source_trial_number:int=20
    output_dir:Path=Path("outputs/results/e5b_feature_selection_ext03_run_01")
    k_values:tuple[int,...]=(5,10,20)

    @staticmethod
    def project_root():
        return Path(__file__).resolve().parents[2]

    def __post_init__(self):
        if not isinstance(self.experiment_id,str) or not self.experiment_id or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in self.experiment_id):
            raise ValueError("experiment_id must be a safe non-empty filename component.")
        if type(self.source_trial_number) is not int or self.source_trial_number<0:
            raise ValueError("source_trial_number must be a nonnegative integer.")
        if not isinstance(self.k_values,(list,tuple)) or not self.k_values or any(type(k) is not int or k<1 for k in self.k_values):
            raise ValueError("k_values must be positive integers.")
        if len(set(self.k_values))!=len(self.k_values):
            raise ValueError("k_values must be unique.")
        object.__setattr__(self,"k_values",tuple(self.k_values))
        for name in ("source_study_dir","output_dir"):
            value=getattr(self,name)
            if not isinstance(value,(str,Path)) or not str(value).strip():
                raise ValueError(f"Invalid {name}.")
            object.__setattr__(self,name,(self.project_root()/Path(value).expanduser()).resolve())

    @classmethod
    def from_json(cls,path,project_root=None):
        root=Path(project_root).resolve() if project_root is not None else cls.project_root()
        raw=json.loads((root/Path(path).expanduser()).read_text(encoding="utf-8"))
        if not isinstance(raw,dict):
            raise ValueError("Expected JSON object.")
        unknown=set(raw)-{f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}.")
        for name,default in (("source_study_dir","outputs/results/e5b_optuna_ext02_run_01"),
                             ("output_dir","outputs/results/e5b_feature_selection_ext03_run_01")):
            raw[name]=(root/Path(raw.get(name,default)).expanduser()).resolve()
        return cls(**raw)

    def to_dict(self):
        return {f.name:str(getattr(self,f.name)) if isinstance(getattr(self,f.name),Path)
                else list(getattr(self,f.name)) if isinstance(getattr(self,f.name),tuple)
                else getattr(self,f.name) for f in fields(self)}
