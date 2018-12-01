import pytest
import warnings
import sys
from polyssifier import Polyssifier
from sklearn.datasets import make_classification

warnings.filterwarnings("ignore", category=DeprecationWarning)

NSAMPLES = 1000
data, label = make_classification(n_samples=NSAMPLES, n_features=50,
                                  n_informative=10, n_redundant=10,
                                  n_repeated=0, n_classes=2,
                                  n_clusters_per_class=2, weights=None,
                                  flip_y=0.01, class_sep=2.0,
                                  hypercube=True, shift=0.0,
                                  scale=1.0, shuffle=True,
                                  random_state=1988)


@pytest.mark.medium
def test_run():
    report = Polyssifier(data, label, n_folds=10, verbose=1,
                         feature_selection=False,
                         save=False, project_name='polyssifier_runtest')
    assert(True)
