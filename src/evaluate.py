# -*- coding: utf-8 -*-
"""
Down beat evaluation using the criterion of MIREX

It is possible to use mir_eval (see how it works)
"""


import numpy as np
import mir_eval


def eval_downbeats(ref, est, trim=True, tolerance=0.1):
    """
    ref, est : time lists 
    `trim` remove 5 first seconds (convention of MIREX)
    
    returns :
    CMLc : float
        Correct metric level, continuous accuracy

    CMLT : float
    Correct metric level, total accuracy (continuity not required)

    AMLc : float
        Any metric level, continuous accuracy

    AMLt : float
        Any metric level, total accuracy (continuity not required)
    """
    
    ref, est = np.asarray(ref, float), np.asarray(est, float)

    if trim:
        ref = mir_eval.beat.trim_beats(ref)
        est = mir_eval.beat.trim_beats(est)

    F = mir_eval.beat.f_measure(
        ref, est,
        f_measure_threshold=tolerance
    )

    cmlc, cmlt, amlc, amlt = mir_eval.beat.continuity(ref, est)

    return F, cmlc, cmlt, amlc, amlt