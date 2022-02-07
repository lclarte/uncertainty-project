
from dataclasses import dataclass, fields
from typing import List

import numpy as np

@dataclass 
class Overlaps:
    rho   : float = float("nan")
    # For BO
    qbo   : float = float("nan")
    # should be used when we do experiments (so q is not exactly m)
    mbo   : float = float("nan")
    # for ERM
    qerm  : float = float("nan")
    m     : float = float("nan")
    V     : float = float("nan")
    # BO - ERM overlap
    Q     : float = float("nan")

    hatqbo: float = float("nan")
    hatqerm: float = float("nan")
    hatm  : float = float("nan")
    hatV  : float = float("nan")
    hatQ  : float = float("nan")

    def get_teacher_bo_erm_covariance(self, sigma : float = 0., add_noise : bool = True) -> np.ndarray:
        teacher_param = self.rho + sigma**2 if add_noise else self.rho
        return np.array([
            [teacher_param, self.qbo, self.m],
            [self.qbo, self.qbo, self.Q],
            [self.m, self.Q, self.qerm]
        ])

    def get_teacher_bo_erm_hat_covariance(self) -> np.ndarray:
        return np.array([
        [1.0,      0     , 0],
        [0  , self.hatqbo     , self.hatQ], 
        [0  , self.hatQ  , self.hatqerm],
    ])

def average_overlaps(overlaps_list : List[Overlaps]) -> Overlaps:
    mean_overlaps = Overlaps()
    for field in fields(mean_overlaps):
        try:
            # ok because we only have float in the overlaps
            mean_value = np.mean([getattr(o, field.name) for o in overlaps_list])
        except:
            mean_value = float('nan')
        setattr(mean_overlaps, field.name, mean_value)
    return mean_overlaps
