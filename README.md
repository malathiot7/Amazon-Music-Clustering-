# Amazon-Music-Clustering-
This project uses unsupervised machine learning to cluster Amazon Music tracks based on audio features. With millions of songs, manually tagging genres or organizing playlists becomes difficult. Clustering helps automatically group similar songs and uncover patterns that may represent genres, moods, or listening behaviors.
A Machine Learning project to automatically group similar songs based on their audio characteristics using **unsupervised learning**.

---

## 📌 Project Overview
This project applies unsupervised machine learning techniques to cluster songs from the Amazon music dataset based on their audio characteristics. With millions of tracks available, genre labeling and playlist organization become challenging. This project uses clustering algorithms to automatically group similar songs — revealing patterns that may represent genres, moods, or listening styles
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

Filename: single_genre_artists.csv

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
- Load dataset using pandas
✔ Explore shape, datatypes, missing values, duplicates
✔ Remove irrelevant columns:
track_name, track_id, artist_name
- Handled missing values  
- Visualized feature distributions  

### 2️⃣ Feature Scaling
Used **StandardScaler** — scaling is critical for distance-based algorithms.

### 3️⃣ Feature Selection
Selected key musical attributes ().danceability, energy, loudness, speechiness, acousticness,
instrumentalness, liveness, valence, tempo, duration_ms
Dimensionality Reduction (Optional)

Used for visualization only:

PCA (2 components) → for 2D scatter plot

t-SNE (optional) for better separation in high-dimensional music space

### 4️⃣ Clustering
- Applied **K-Means**
- Determined best number of clusters using:
  - Elbow Method
  - Silhouette Score
  - 
Cluster Evaluation & Interpretation

Use:
     Silhouette Score
     Davies-Bouldin Index
     Cluster centroids interpret meanings:

### 5️⃣ Visualization
- PCA 2D scatter plot  
- Heatmaps per cluster  
- Feature distribution comparisons  

---

## 📊 Results & Interpretation

The final model grouped songs into meaningful patterns:

| Cluster ID | Characteristics                | Cluster Meaning      |
| ---------- | ------------------------------ | -------------------- |
| 0          | High energy + fast tempo       | 🎉 Party / Dance     |
| 1          | Low energy + high acousticness | 🎧 Chill / Acoustic  |
| 2          | High speechiness               | 🎤 Rap / Spoken word |


These groupings can support playlist automation, recommendation engines, or audience behavior analysis.

---

## 🖥️ Streamlit App (Optional UI)

A user-friendly Streamlit dashboard is included to:

- Upload datasets  
- Run clustering interactively  
- View metrics and visualizations  
- Download clustered output  

Launch with:

streamlit run app.py
