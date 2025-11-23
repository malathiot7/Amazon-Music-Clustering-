# Amazon-Music-Clustering-
This project uses unsupervised machine learning to cluster Amazon Music tracks based on audio features. With millions of songs, manually tagging genres or organizing playlists becomes difficult. Clustering helps automatically group similar songs and uncover patterns that may represent genres, moods, or listening behaviors.
# 🎵 Amazon Music Clustering
A Machine Learning project to automatically group similar songs based on their audio characteristics using **unsupervised learning**.

---

## 📌 Project Overview
With millions of songs online, manually tagging or grouping tracks into genres is inefficient.  
This project uses clustering algorithms — primarily **K-Means** — to automatically identify patterns and group similar tracks based on audio features like:

- Danceability  
- Energy  
- Tempo  
- Loudness  
- Acousticness  
- Speechiness  
- Instrumentalness  
- Valence  

The output clusters may represent genres, moods, or listening behavior patterns.

---

## 🎯 Project Goals

✔ Clean and explore the dataset  
✔ Scale and prepare features  
✔ Apply clustering (K-Means, optional DBSCAN)  
✔ Determine optimal cluster count using Elbow Method & Silhouette Score  
✔ Visualize clusters using PCA  
✔ Interpret clusters for real-world meaning  
✔ Build a simple Streamlit UI for user interaction  

---

## 🧠 Skills & Tools Used

| Category | Technologies |
|---------|-------------|
| Language | Python |
| Libraries | pandas, numpy, scikit-learn, seaborn, matplotlib |
| ML Techniques | K-Means, Silhouette Score, Elbow Method, PCA |
| Deployment | Streamlit |
| Domain | Music Analytics & Recommendation Systems |

---

## 📂 Dataset

The dataset contains various numeric audio features extracted from Amazon Music.  
Example columns:

| Feature | Description |
|---------|------------|
| `danceability` | How suitable the track is for dancing |
| `energy` | Measure of intensity & activity |
| `valence` | Musical positivity score |
| `tempo` | Beats per minute |
| `instrumentalness` | Likelihood the track is instrumental |

Non-numeric identifiers such as track name and artist name were removed before clustering.

---

## ⚙️ Methodology

### 1️⃣ Data Exploration & Cleaning
- Removed duplicates and irrelevant columns  
- Handled missing values  
- Visualized feature distributions  

### 2️⃣ Feature Scaling
Used **StandardScaler** — scaling is critical for distance-based algorithms.

### 3️⃣ Feature Selection
Selected key musical attributes (e.g., `energy`, `tempo`, `valence`).

### 4️⃣ Clustering
- Applied **K-Means**
- Determined best number of clusters using:
  - Elbow Method
  - Silhouette Score

### 5️⃣ Visualization
- PCA 2D scatter plot  
- Heatmaps per cluster  
- Feature distribution comparisons  

---

## 📊 Results & Interpretation

The final model grouped songs into meaningful patterns:

| Cluster | Characteristics | Interpretation |
|--------|-----------------|----------------|
| Cluster 0 | High tempo, high energy | Workout / EDM |
| Cluster 1 | Calm tempo, high acousticness | Chill / Relax |
| Cluster 2 | High speechiness | Hip-hop / Rap |
| Cluster 3 | High valence + danceability | Pop / Happy vibes |

These groupings can support playlist automation, recommendation engines, or audience behavior analysis.

---

## 🖥️ Streamlit App (Optional UI)

A user-friendly Streamlit dashboard is included to:

- Upload datasets  
- Run clustering interactively  
- View metrics and visualizations  
- Download clustered output  

Launch with:

```bash
streamlit run app.py
