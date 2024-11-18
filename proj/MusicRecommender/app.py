from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Load all datasets
data = pd.read_csv('data.csv')
year_data = pd.read_csv('data_by_year.csv')

# Define the numerical columns used for clustering and recommendation
number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
               'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

# Create and fit the clustering pipeline for the main dataset
song_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=20, verbose=False))
], verbose=False)

# Fit the clustering pipeline to the data
X = data[number_cols]
song_cluster_pipeline.fit(X)
data['cluster_label'] = song_cluster_pipeline.predict(X)

app = Flask(__name__)

# Function to get song data
def get_song_data(song, spotify_data):
    # Case insensitive lookup for the song
    song_data = spotify_data[(spotify_data['name'].str.lower() == song['name'].lower()) & (spotify_data['year'] == song['year'])]

    if song_data.empty:
        print(f"Warning: Song '{song['name']}' from year {song['year']} not found in the dataset.")
        return None

    return song_data.iloc[0]

# Function to get the mean vector for a list of songs
def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)

        if song_data is None:
            continue

        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)

    if len(song_vectors) == 0:
        raise ValueError("No valid songs found in the dataset for the given input.")

    song_matrix = np.array(song_vectors)
    return np.mean(song_matrix, axis=0)

# Function to recommend songs based on song list
def recommend_songs(song_list, spotify_data, song_cluster_pipeline, n_songs=10):
    metadata_cols = ['name', 'year', 'artists']

    if song_list:
        try:
            song_center = get_mean_vector(song_list, spotify_data)
        except ValueError:
            print("No valid songs found in the dataset for the given input.")
            return []

        scaler = song_cluster_pipeline.steps[0][1]
        scaled_data = scaler.transform(spotify_data[number_cols])
        scaled_song_center = scaler.transform(song_center.reshape(1, -1))
        distances = cdist(scaled_song_center, scaled_data, 'cosine')
        index = list(np.argsort(distances)[:, :n_songs][0])
    else:
        # If no specific song is provided, just return random recommendations from the dataset
        index = spotify_data.sample(n=n_songs).index

    rec_songs = spotify_data.iloc[index]
    # Format the recommendation results as a list of dictionaries with desired structure
    recommendations = []
    for _, row in rec_songs.iterrows():
        recommendations.append({
            'name': row['name'],
            'year': row['year'],
            'artists': row['artists']
        })

    return recommendations

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = None
    if request.method == 'POST':
        song_name = request.form.get('song_name', '').strip()
        song_year = request.form.get('song_year', '').strip()

        song_list = []
        if song_name and song_year:
            try:
                song_year = int(song_year)
                song_list = [{'name': song_name, 'year': song_year}]
            except ValueError:
                print(f"Invalid year: {song_year}")

        if song_list:
            recommendations = recommend_songs(song_list, data, song_cluster_pipeline, n_songs=5)

    return render_template('home.html', recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)