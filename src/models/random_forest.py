from sklearn.ensemble import RandomForestRegressor


def build_random_forest(seed: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
