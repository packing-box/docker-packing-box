# -*- coding: UTF-8 -*-
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from contextlib import suppress
from functools import wraps
from math import ceil
from matplotlib.colors import ListedColormap
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.decomposition import FastICA, PCA
from sklearn.impute import SimpleImputer
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import export_text, plot_tree

from .algorithm import Algorithm 


__all__ = ["VISUALIZATIONS"]


cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])


def _preprocess(f):
    """ This decorator preprocesses the input data and updates the keyword-arguments with this fitted data. """
    @wraps(f)
    def _wrapper(*args, **kwargs):
        X = kwargs['data']
        n_cols = len(X.columns)
        n, p = kwargs.get('n_components', min(20, n_cols)), kwargs.get('perplexity', 30)
        suffix = ""
        X = SimpleImputer(missing_values=np.nan, strategy=kwargs.get('imputer_strategy', "mean")).fit_transform(X)
        X = MinMaxScaler().fit_transform(X)
        # Update the keyword-arguments with the scaled data, since it may be used in the visualization
        kwargs['scaled_data'] = X 
        # preprocess data with a PCA with n components to reduce the high dimensionality (better performance)
        if n < n_cols:
            ra = kwargs.get('reduction_algorithm', "PCA")
            a = {'ICA': FastICA, 'PCA': PCA}[ra](n, random_state=42)
            suffix += "_%s%d" % (ra.lower(), n)
            if 'target' in kwargs:
                a.fit(X, kwargs['target'])
                X = a.transform(X)
            else:
                X = a.fit_transform(X)
        # now reduce the n components to 2 dimensions with t-SNE (better results but less performance) if relevant
        if n > 2:
            X = TSNE(2, random_state=42, perplexity=p).fit_transform(X)
            suffix += "_tsne2-p%d" % p
        kwargs['reduced_data'] = X
        fig = f(*args, **kwargs)
        fig.dst_suffix = suffix
        return fig
    return _wrapper


def image_dt(classifier, width=5, fontsize=10, **params):
    params['filled'] = True
    fig = plt.figure()
    plot_tree(classifier, **params)
    return fig


@_preprocess
def image_knn(classifier, **params):
    X, y = params['data'], params['target']
    # retrain kNN with the preprocessed data (with dimensionality reduced to N=2, hence not using 'classifier')
    knn = KNeighborsClassifier(**params['algo_params'])
    knn.fit(X, y)
    # now set color map then plot
    labels = list(y.label.unique())
    colors = mpl.cm.get_cmap("jet", len(labels))
    fig, axes = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(knn, X, cmap=colors, ax=axes, alpha=.3,
                                           response_method="predict", plot_method="pcolormesh", shading="auto")
    plt.scatter(X[:, 0], X[:, 1], c=[labels.index(v) for v in y.label.ravel()][::-1], cmap=colors, alpha=1.0)
    return fig


def image_rf(classifier, width=5, fontsize=10, **params):
    n = len(classifier.estimators_)
    rows = ceil(n / width)
    cols = width if rows > 1 else n
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=tuple(map(lambda x: 2*x, (cols, rows))), dpi=900)
    # flatten axes, otherwise it is a matrix of all subplots
    with suppress(TypeError):
        axes = [ax for lst in axes for ax in lst]
    params['filled'] = True
    for i in range(n):
        plot_tree(classifier.estimators_[i], ax=axes[i], **params)
        axes[i].set_title("Estimator: %d" % i, fontsize=fontsize)
    return fig


@_preprocess
def image_clustering(classifier, **params):
    X, y = params['data'], params['target']
    X_reduced = params['reduced_data']  
    X_scaled = params['scaled_data']      
    # retrain either with the preprocessing data or with the original data
    cls = Algorithm.get(params['algo_name']).base(**params['algo_params'])
    if params.get('reduce_train_data', False):
        label = cls.fit_predict(X_reduced)
    else : 
        label = cls.fit_predict(X_scaled)
    # Get labels for hierarchical clustering
    is_hierarchical = isinstance(cls, AgglomerativeClustering)
    if is_hierarchical: 
        if params.get('reduce_train_data', False):
            Z = linkage(X_reduced, method='ward')
        else : 
            Z = linkage(X_scaled, method='ward')
        label = fcluster(Z, t=cls.n_clusters_, criterion='maxclust')
    # now set color map
    colors = mpl.cm.get_cmap("jet")
    # Adjust number of plots and size of figure
    features = params['features'][0]
    n_features = len(features)
    n_plots = 2 + int(is_hierarchical) + int(params['plot_extensions']) + int(params['plot_formats']) + n_features
    fig, axes = plt.subplots(n_plots ,figsize=(10 , 6 + 2 * n_plots))
    current_ax = 0 
    # Select zone of data to plot
    if params['range'] is not None:
        if is_hierarchical:
            raise ValueError("Error: Zone selection is not compatible with hierarchical clustering")
        x_min, x_max, y_min, y_max = params['range']
        range_mask = (X_reduced[:, 0] >= x_min) & (X_reduced[:, 0] <= x_max) & (X_reduced[:, 1] >= y_min) & (X_reduced[:, 1] <= y_max)
        X_reduced = X_reduced[range_mask]
        X = X[range_mask]
        label = label[range_mask]
        params['labels'] = params['labels'][range_mask]
        params['extension'] = params['extension'][range_mask]
        params['format'] =  params['format'][range_mask]
        if X_reduced.size == 0 :
            raise ValueError("Error: The selected zone is empty")
    # Plot dendrogram for hierarchical clustering
    if is_hierarchical: 
        d = dendrogram(Z, ax=axes[0], no_plot= True)
        dendrogram(Z, ax=axes[0], truncate_mode='level', p=params['hierarchy_levels'], leaf_rotation=90., leaf_font_size=8., show_contracted=params['hierarchy_levels'] <= 5)        
        axes[current_ax].set_title("Dendrogram")
        current_ax += 1
    # Plot predicted cluster labels
    if is_hierarchical:
        axes[current_ax].scatter(X_reduced[d['leaves'],0],X_reduced[d['leaves'],1], color=d['leaves_color_list'])
    else : 
        predicted_labels = np.unique(label)
        print(predicted_labels)
        cluster_colors = {i: colors(i / len(predicted_labels)) for i in predicted_labels} if not is_hierarchical else cluster_colors
        for i in predicted_labels:
            axes[current_ax].scatter(X_reduced[label == i, 0], X_reduced[label == i, 1] , label=i, color=cluster_colors[i])
    axes[current_ax].set_title("Clusters")
    current_ax += 1 
    # Plot true labels
    if params['multiclass']:
        y_labels = np.unique(params['labels'])
        colors_labels = mpl.cm.get_cmap("jet", len(y_labels))
        label_map = {label: 'Not packed' if label == '-' else f'Packed : {label}' for label in y_labels}
    else : 
        colors_labels = mpl.cm.get_cmap("jet", 2)
        y_labels = np.unique(y.label.ravel())
        label_map = {0: 'Not packed', 1: 'Packed'}
    for i, y_label in enumerate(y_labels):
        labels_mask = params['labels'] == y_label if params['multiclass'] else y.label.ravel() == y_label
        axes[current_ax].scatter(X_reduced[labels_mask, 0], X_reduced[labels_mask, 1],
                        label=label_map[y_label], color=colors_labels(i), alpha=1.0)
    axes[current_ax].legend(loc='upper left', bbox_to_anchor=(1, 1))  
    axes[current_ax].set_title("Target")
    current_ax += 1
    # Plot file formats
    if params['plot_formats']:
        unique_formats = np.unique(params['format'])
        for file_format in unique_formats:
            format_mask = params['format'] == file_format
            axes[current_ax].scatter(X_reduced[format_mask, 0], X_reduced[format_mask, 1],
                            label=file_format, cmap=colors, alpha=1.0)
        axes[current_ax].legend(loc='upper left', bbox_to_anchor=(1, 1))
        axes[current_ax].set_title("File Formats")
        current_ax += 1 
    # Plot file extensions
    if params['plot_extensions']:
        unique_extensions = np.unique(params['extension'])
        for file_extension in unique_extensions:
            extension_mask = params['extension'] == file_extension
            axes[current_ax].scatter(X_reduced[extension_mask, 0], X_reduced[extension_mask, 1],
                            label=file_extension, cmap=colors, alpha=1.0)
        axes[current_ax].legend(loc='upper left', bbox_to_anchor=(1, 1))
        axes[current_ax].set_title("File Extensions")
        current_ax += 1 
    # Plot selected features
    if features :
        for i, feature in enumerate(features) : 
            unique_feature_values = np.unique(X[feature])
            # Plot a continous colorbar if the feature is not boolean and a legend otherwise 
            if len(unique_feature_values) > 2:
                axes[current_ax].scatter(X_reduced[:, 0], X_reduced[:, 1], c=X[feature].to_numpy(), cmap=colors, alpha=1.0)
                norm = mpl.colors.Normalize(vmin=X[feature].min(), vmax=X[feature].max())
                # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colors), ax=axes[n_plots - n_features +i],
                # orientation='vertical')
                colorbar_ax = fig.add_axes([axes[current_ax].get_position().x1 + 0.01,
                            axes[current_ax].get_position().y0 - 0.08,
                            0.01,
                            axes[current_ax].get_position().height])
                fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colors), cax=colorbar_ax, orientation='vertical')
            else : 
                label_map = {0: 'False', 1: 'True'}
                for feature_value in unique_feature_values:
                    axes[current_ax].scatter(X_reduced[X[feature] == feature_value, 0], X_reduced[X[feature] == feature_value, 1],
                                        label=label_map[feature_value], cmap=colors, alpha=1.0)
                    axes[current_ax].legend(loc='upper left', bbox_to_anchor=(1, 1))  
            axes[current_ax].set_title(feature)
            current_ax += 1 
    title = generate_title(params)
    plt.suptitle(title, fontweight="bold", fontsize=14, y=1.01)
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    return fig

def generate_title(params):
    algo_name = params['algo_name']
    dataset_name = params['dataset_name']
    n_components = params.get('n_components', None)
    reduction_algorithm = params.get('reduction_algorithm', None)
    perplexity = params.get('perplexity', None)
    dimensionality_reduction_info = ""
    if n_components is not None and reduction_algorithm is not None:
        dimensionality_reduction_info = f"{reduction_algorithm} ({n_components} Components)"
        if reduction_algorithm == "PCA" and n_components > 2:
            dimensionality_reduction_info += f" and t-SNE (2 Components, Perplexity: {perplexity})"
    elif n_components is not None:
        dimensionality_reduction_info = f"Dimensionality Reduction ({n_components} Components)"
    title = f"{algo_name} Visualization of dataset {dataset_name} \n with {dimensionality_reduction_info}"
    return title

def text_dt(classifier, **params):
    return export_text(classifier, **params)


def text_rf(classifier, **params):
    s = ""
    for i in range(len(classifier.estimators_)):
        s += "\nEstimator: %d\n" % i
        s += export_text(classifier.estimators_[i], **params)
    return s


VISUALIZATIONS = {
    'DT':  {'image': image_dt, 'text': text_dt},
    'kNN': {'image': image_knn, 'data': True},
    'RF':  {'image': image_rf, 'text': text_rf},
}
for a in ['AC', 'AP', 'Birch', 'DBSCAN', 'KMeans', 'MBKMeans', 'MS', 'OPTICS', 'SC']:
    VISUALIZATIONS[a] = {'image': image_clustering, 'data': True, 'target': True}

