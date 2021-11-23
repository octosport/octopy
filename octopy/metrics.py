import numpy as np
import pandas as pd

def check_inputs(probabilities,true_results):
    '''Check metrics inputs'''
    assert probabilities.shape[0]==true_results.shape[0],'The number of match prediction and result should be the same'
    assert np.alltrue(probabilities>0), 'probabilities should be larger than 0.'
    assert np.alltrue(probabilities<=1), 'Probabilities should be lower or equal to 1.'
    assert np.alltrue( abs(1-probabilities.sum(1))<1e-3), 'Probabilities should sum to 1.'
    assert np.alltrue(~probabilities.isin([np.inf, -np.inf,np.nan]))
    assert probabilities.shape[1]==3, 'You must provide probabilities for 1, 2 and X.'
    assert np.alltrue([x in ['1','2','X'] for x in probabilities.columns]),'Probabilities columns name have to be "1","2" and "X" '
    assert np.alltrue([x in ['1', '2', 'X'] for x in np.unique(true_results)])
    assert isinstance(probabilities,pd.DataFrame)
    assert isinstance(true_results,pd.Series)

def compute_1x2_log_loss(probabilities, true_results):
    '''
    Compute the log-loss for 1x2 football results.

    Parameters
    ----------
    probabilities: pd.DataFrame
        A dataframe that contains probabilities for 1, X ,and 2 results.

    true_results: pd.Series
        A pd.Series that contains the true results, 1 X or 2.

    Returns
    -------
    The log-loss

    '''

    check_inputs(probabilities,true_results)
    true_results_dum = pd.get_dummies(true_results)
    return (np.log(probabilities)*true_results_dum).sum(1).mean()


def compute_1x2_hit_ratio(probabilities, true_results):
    '''
    Compute the hit-ratio for 1x2 football results.

    Parameters
    ----------
    probabilities: pd.DataFrame
        A dataframe that contains probabilities for 1, X ,and 2 results.

    true_results: pd.Series
        A pd.Series that contains the true results, 1 X or 2.

    Returns
    -------
    The hit-ratio

    '''

    check_inputs(probabilities, true_results)
    predicted_results = probabilities.idxmax(1)
    return (predicted_results==true_results).mean()