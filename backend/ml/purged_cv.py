"""Purged K-Fold Cross-Validation with Embargo.

AFML Ch.7: Prevents information leakage in time-series cross-validation
by purging training observations whose labels overlap with the test set,
and adding an embargo buffer after each test fold.
"""
import numpy as np
from sklearn.model_selection import BaseCrossValidator


class PurgedKFoldCV(BaseCrossValidator):
    """K-Fold CV with purging and embargo for overlapping labels.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    label_ends : np.ndarray
        Array of label end indices (one per sample). label_ends[i] = index of
        last bar used by label i. Used to determine overlap with test set.
    embargo_pct : float
        Fraction of total samples to embargo after each test set.
    """

    def __init__(self, n_splits: int = 5, label_ends: np.ndarray | None = None,
                 embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.label_ends = label_ends
        self.embargo_pct = embargo_pct

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        indices = np.arange(n_samples)
        embargo_size = int(n_samples * self.embargo_pct)

        # Create temporal folds (not random — preserve time order)
        fold_size = n_samples // self.n_splits
        folds = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = start + fold_size if i < self.n_splits - 1 else n_samples
            folds.append(indices[start:end])

        for test_fold_idx in range(self.n_splits):
            test_indices = folds[test_fold_idx]
            test_start = test_indices[0]
            test_end = test_indices[-1]

            # Build train indices: all folds except test
            train_indices = np.concatenate(
                [folds[i] for i in range(self.n_splits) if i != test_fold_idx]
            )

            # Purge: remove training samples whose labels overlap with test period
            if self.label_ends is not None:
                purge_mask = np.ones(len(train_indices), dtype=bool)
                for j, train_idx in enumerate(train_indices):
                    label_end = self.label_ends[train_idx]
                    # If this training sample's label extends into the test period
                    if label_end >= test_start and train_idx < test_start:
                        purge_mask[j] = False
                train_indices = train_indices[purge_mask]

            # Embargo: remove training samples immediately after the test set
            if embargo_size > 0:
                embargo_start = test_end + 1
                embargo_end = min(test_end + embargo_size, n_samples - 1)
                embargo_mask = ~(
                    (train_indices >= embargo_start) &
                    (train_indices <= embargo_end)
                )
                train_indices = train_indices[embargo_mask]

            yield train_indices, test_indices
