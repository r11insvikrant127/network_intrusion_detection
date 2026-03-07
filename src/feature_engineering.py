import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


def feature_importance(X, y):

    model = RandomForestClassifier()
    model.fit(X, y)

    importance = model.feature_importances_

    importance_percent = importance * 100

    return importance_percent


import matplotlib.pyplot as plt
import seaborn as sns

def correlation_heatmap(df):

    numeric_df = df.select_dtypes(include=['number'])

    plt.figure(figsize=(12,8))

    sns.heatmap(numeric_df.corr(), cmap="coolwarm")

    plt.title("Correlation Heatmap")

    plt.savefig("outputs/correlation_heatmap.png")

    plt.close()


def apply_pca(X, n_components=20):

    pca = PCA(n_components=n_components)

    X_pca = pca.fit_transform(X)

    return X_pca, pca