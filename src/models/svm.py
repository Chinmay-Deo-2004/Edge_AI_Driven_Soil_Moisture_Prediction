from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_svm(C: float = 1.0, gamma: str = "scale", pca_components: int | None = None) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if pca_components:
        from sklearn.decomposition import PCA
        steps.append(("pca", PCA(n_components=pca_components, random_state=42)))
    steps.append(("svm", SVC(kernel="rbf", C=C, gamma=gamma)))
    return Pipeline(steps)
