from __future__ import annotations

from sklearn.model_selection import StratifiedKFold, cross_val_score


def macro_f1_cross_validation(model, X, y, folds: int = 5) -> list[float]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    return cross_val_score(model, X, y, cv=cv, scoring="f1_macro").tolist()
