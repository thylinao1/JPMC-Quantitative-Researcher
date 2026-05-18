"""Operational profile for a credit-risk decision rule.

A profit number on its own does not communicate the trade-off. A
risk officer wants to know: how many loans were rejected, how many
of those would have been good, what is the approved-loan volume,
and what is the default rate on the approved book. These are the
numbers that make a profit headline auditable.
"""

from __future__ import annotations

import numpy as np


def operational_profile(probs, labels, threshold, loan_amount):
    """Return the operational view of a threshold decision."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    predictions = (probs >= threshold).astype(int)
    n = int(labels.shape[0])
    rejected = int(predictions.sum())
    approved = int(n - rejected)
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    approved_defaults = fn
    approved_volume = approved * loan_amount
    return {
        "n": n,
        "rejected": rejected,
        "approved": approved,
        "rejection_rate": float(rejected / n) if n else float("nan"),
        "false_rejection_count": fp,
        "approved_volume": float(approved_volume),
        "approved_default_rate": float(approved_defaults / approved) if approved else float("nan"),
    }
