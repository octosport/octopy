import numpy as np
import pandas as pd
from octopy.metrics import compute_1x2_log_loss, compute_1x2_hit_ratio


def test_log_loss():
    p = pd.DataFrame(np.eye(3)*0.97+0.01,columns=['1','2','X'])
    t = pd.Series(['1','2','X'])
    np.testing.assert_almost_equal(np.log(0.98), compute_1x2_log_loss(p,t))

def test_hit_ratio():
    p = pd.DataFrame(np.eye(3)*0.97+0.01,columns=['1','2','X'])
    t = pd.Series(['1','2','X'])
    np.testing.assert_almost_equal(1, compute_1x2_hit_ratio(p,t))