# Kaggle Toolbox

Koolbox is a collection of helper functions and utilities designed to simplify training  machine learning models in Kaggle competitions. This library abstracts away repetitive boilerplate code, allowing competitors to focus on more important tasks.

## Installation

```bash
pip install koolbox
```

## Usage

### Trainer

```python
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from koolbox import Trainer


X = pd.DataFrame(...)
y = pd.Series(...)

trainer = Trainer(
    estimator=RandomForestClassifier(random_state=42),
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    metric=roc_auc_score,
    task="binary",
    verbose=True
)

trainer.fit(X, y)

X_test = pd.DataFrame(...)
preds = trainer.predict(X_test)

oof_preds = trainer.oof_preds
overall_score = trainer.overall_score
fold_scores = trainer.fold_scores
```

### SequentialFeatureSelector

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import pandas as pd

from koolbox import SequentialFeatureSelector


X = pd.DataFrame(...)
y = pd.Series(...)
X_test = pd.DataFrame(...)

sfs = SequentialFeatureSelector(
    Ridge(),
    cv=KFold(n_splits=5, random_state=42, shuffle=True),
    objective="minimize",
    direction="backward",
    metric=root_mean_squared_error
)

X = sfs.fit_transform(X, y)
X_test = sfs.transform(X_test)

selected_features = sfs.selected_features
```