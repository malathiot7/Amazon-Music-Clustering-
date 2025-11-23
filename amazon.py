# streamlit_visual_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

@st.cache_data
def scale_features(df, features):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features].values)
    return scaled, scaler

def compute_kmeans(scaled, k, random_state=42):
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(scaled)
    metrics = {}
    try:
        metrics['silhouette'] = silhouette_score(scaled, labels)
    except Exception:
        metrics['silhouette'] = None
    try:
        metrics['davies_bouldin'] = davies_bouldin_score(scaled, labels)
    except Exception:
        metrics['davies_bouldin'] = None
    metrics['inertia'] = float(model.inertia_)
    return labels, model, metrics

def compute_dbscan(scaled, eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(scaled)
    metrics = {}
    if len(set(labels)) > 1 and (len(set(labels)) != 1 or -1 not in labels):
        try:
            metrics['silhouette'] = silhouette_score(scaled, labels)
        except Exception:
            metrics['silhouette'] = None
        try:
            metrics['davies_bouldin'] = davies_bouldin_score(scaled, labels)
        except Exception:
            metrics['davies_bouldin'] = None
    else:
        metrics['silhouette'] = None
        metrics['davies_bouldin'] = None
    return labels, model, metrics

def plot_pca(scaled, labels, title="PCA 2D"):
    pca = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(scaled)
    fig, ax = plt.subplots(figsize=(7,5))
    sc = ax.scatter(comps[:,0], comps[:,1], c=labels, cmap='tab10', s=40, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    plt.colorbar(sc, ax=ax, pad=0.02, label='cluster')
    st.pyplot(fig)

def plot_tsne(scaled, labels, perplexity=30):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
    comps = tsne.fit_transform(scaled)
    fig, ax = plt.subplots(figsize=(7,5))
    sc = ax.scatter(comps[:,0], comps[:,1], c=labels, cmap='tab10', s=40, alpha=0.8)
    ax.set_title("t-SNE 2D")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.colorbar(sc, ax=ax, pad=0.02, label='cluster')
    st.pyplot(fig)

def plot_feature_bar(cluster_means):
    fig, ax = plt.subplots(figsize=(10,5))
    cluster_means.T.plot(kind='bar', ax=ax)
    ax.set_ylabel("Average feature value")
    ax.set_title("Average Features per Cluster")
    ax.legend(title="Cluster")
    st.pyplot(fig)

def plot_heatmap(cluster_means):
    fig, ax = plt.subplots(figsize=(9,5))
    sns.heatmap(cluster_means, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title("Cluster Feature Heatmap")
    st.pyplot(fig)

def kde_distributions(df, features, label_col='cluster'):
    for f in features:
        fig, ax = plt.subplots(figsize=(8,3.5))
        sns.kdeplot(data=df, x=f, hue=label_col, fill=True, alpha=0.4, ax=ax)
        ax.set_title(f"Distribution of {f} by Cluster")
        st.pyplot(fig)

# -------------------------
# App layout
# -------------------------
st.set_page_config(layout="wide", page_title="Music Clustering Dashboard")
st.title("🎵 Music Clustering Dashboard (KMeans / DBSCAN)")

with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader("Upload single_genre_artists.csv", type=['csv'])
    sample_mode = st.checkbox("Use sample data (if uploaded file missing)", value=False)
    st.markdown("---")
    st.subheader("Feature selection")
    default_features = [
        'danceability','energy','loudness','speechiness','acousticness',
        'instrumentalness','liveness','valence','tempo','duration_ms'
    ]
    features = st.multiselect("Select features for clustering", default_features, default=default_features)
    st.markdown("---")
    st.subheader("Algorithm settings")
    algorithm = st.selectbox("Clustering algorithm", ["KMeans", "DBSCAN"])
    if algorithm == "KMeans":
        k = st.slider("Number of clusters (K)", 2, 12, 3)
    else:
        eps = st.slider("DBSCAN eps", 0.1, 5.0, 0.6)
        min_samples = st.slider("DBSCAN min_samples", 1, 30, 5)
    st.markdown("---")
    st.subheader("Visualization")
    show_pca = st.checkbox("Show PCA plot", value=True)
    show_tsne = st.checkbox("Show t-SNE (slow)", value=False)
    tsne_perp = st.slider("t-SNE perplexity", 5, 50, 30) if show_tsne else None
    st.markdown("---")
    run_btn = st.button("Run Clustering")

# -------------------------
# Main: load data
# -------------------------
if uploaded:
    try:
        df = load_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.stop()
elif sample_mode:
    st.info("Sample mode: generating synthetic demo data (10 clusters-ish)")
    # create small synthetic sample if user chooses sample mode
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'track_name': [f"Track {i}" for i in range(n)],
        'artist_name': [f"Artist {i%50}" for i in range(n)],
        'danceability': np.clip(np.random.rand(n) + np.random.randn(n)*0.05, 0, 1),
        'energy': np.clip(np.random.rand(n) + np.random.randn(n)*0.05, 0, 1),
        'loudness': np.random.randn(n)*6 - 8,
        'speechiness': np.clip(np.random.rand(n)*0.5, 0, 1),
        'acousticness': np.clip(np.random.rand(n)*0.8, 0, 1),
        'instrumentalness': np.clip(np.random.rand(n)*0.6, 0, 1),
        'liveness': np.clip(np.random.rand(n)*0.7, 0, 1),
        'valence': np.clip(np.random.rand(n), 0, 1),
        'tempo': np.random.normal(120, 15, n).clip(40, 220),
        'duration_ms': np.random.normal(210000, 30000, n).clip(60000, 600000).astype(int)
    })
else:
    st.info("Upload a CSV file from the sidebar to begin, or enable sample mode.")
    st.stop()

# validate features exist
missing = [f for f in features if f not in df.columns]
if missing:
    st.error(f"The following selected features are missing from the dataset: {missing}")
    st.stop()

# Remove unwanted columns for numeric processing but keep originals for display
display_df = df.copy()

# Optionally drop identifiers before scaling (we keep them in display_df)
cols_to_drop = ['track_name', 'artist_name', 'track_id']
proc_df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

# Select numeric subset (or explicit features)
numeric_df = proc_df[features].select_dtypes(include=[np.number]).copy()

if numeric_df.shape[1] == 0:
    st.error("No numeric feature columns found to cluster on. Please pick numeric features.")
    st.stop()

# Scale
scaled, scaler = scale_features(numeric_df, numeric_df.columns.tolist())

# Run clustering on button click
if run_btn:
    if algorithm == "KMeans":
        labels, model, metrics = compute_kmeans(scaled, k)
        display_df['cluster'] = labels
    else:
        labels, model, metrics = compute_dbscan(scaled, eps, min_samples)
        display_df['cluster'] = labels

    # Show metrics
    st.subheader("Clustering Metrics")
    st.write(metrics)

    # Cluster sizes
    st.subheader("Cluster Sizes")
    ccounts = pd.Series(display_df['cluster']).value_counts().sort_index()
    st.bar_chart(ccounts)

    # PCA visualization
    if show_pca:
        st.subheader("PCA Scatter Plot")
        plot_pca(scaled, display_df['cluster'].fillna(-1).values, title=f"PCA - {algorithm}")

    # t-SNE visualization (slow)
    if show_tsne:
        st.subheader("t-SNE Scatter Plot")
        with st.spinner("Running t-SNE (this may take a while)..."):
            plot_tsne(scaled, display_df['cluster'].fillna(-1).values, perplexity=tsne_perp)

    # Cluster feature means
    st.subheader("Cluster Feature Means")
    cluster_means = display_df.groupby('cluster')[features].mean().sort_index()
    st.dataframe(cluster_means.style.format("{:.3f}"))

    # Bar chart
    plot_feature_bar(cluster_means)

    # Heatmap
    plot_heatmap(cluster_means)

    # Top tracks per cluster (if names exist)
    st.subheader("Top tracks per cluster (sample)")
    for c in sorted(display_df['cluster'].unique()):
        st.markdown(f"**Cluster {c}** ({int((display_df['cluster'] == c).sum())} tracks)")
        cols_show = []
        if 'track_name' in display_df.columns:
            cols_show.append('track_name')
        if 'artist_name' in display_df.columns:
            cols_show.append('artist_name')
        if len(cols_show) == 0:
            cols_show = features[:4]  # show some features
        st.table(display_df[display_df['cluster'] == c][cols_show].head(6))

    # Distribution plots
    st.subheader("Feature Distributions by Cluster")
    kde_distributions(display_df, features, label_col='cluster')

    # Export clustered CSV
    st.subheader("Export")
    csv_bytes = display_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download clustered CSV", csv_bytes, file_name="clustered_music.csv", mime="text/csv")

