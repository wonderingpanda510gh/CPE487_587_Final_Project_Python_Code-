import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import os
from pathlib import Path
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, mean_squared_error, r2_score

from sklearn.decomposition import PCA

def create_dir(path_str):
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path

def evaluate_clustering(latent_space, labels, output_dir="results"):
    out_path = create_dir(output_dir)
    
    # compute silhouette_score, davies_bouldin_score, calinski_harabasz_score
    sil = silhouette_score(latent_space, labels)
    db = davies_bouldin_score(latent_space, labels)
    ch = calinski_harabasz_score(latent_space, labels)
    
    # store the clustering metrics in a DataFrame and save to CSV
    cluster_metrics = pl.DataFrame({
        "metric": ["Silhouette", "Davies_Bouldin", "Calinski_Harabasz"],
        "value": [sil, db, ch]
    })
    cluster_metrics.write_csv(out_path / "clustering_metrics.csv")
    
    # here we use PCA to reduce the dimensional of the latent space to 2D for visualization, and we color the points based on their cluster labels
    pca = PCA(n_components=2)
    components = pca.fit_transform(latent_space)
    
    plt.figure()
    sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=labels, palette='viridis', s=60)
    plt.title('Cluster Visualization')
    save_path = os.path.join(out_path, "cluster_pca_plot.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    return sil, db, ch

def evaluate_regression(y_true, y_pred, target_names, output_dir="results"):
    out_path = create_dir(output_dir)
    results_list = []
    
    for i, name in enumerate(target_names):
        true = y_true[:, i]
        pred = y_pred[:, i]
        
        r2 = r2_score(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        results_list.append({"target": name, "R2": r2, "RMSE": rmse})
        
        plt.figure()
        sns.regplot(x=true, y=pred, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
        plt.title(f'Regression: {name} ($R^2$: {r2:.3f})')
        save_path = os.path.join(out_path, f"prediction_{name}_fit.pdf")
        plt.savefig(save_path, bbox_inches='tight') 
        plt.close()

    # store the regression metrics in a DataFrame and save to CSV
    reg_metrics = pl.DataFrame(results_list)
    reg_metrics.write_csv(out_path / "regression_metrics.csv")