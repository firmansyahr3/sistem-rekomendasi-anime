import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from io import StringIO

# ========================
# LOAD DATA
# ========================
anime = pd.read_csv("C:/Users/DELL/Documents/New folder/anime.csv")
rating = pd.read_csv("C:/Users/DELL/Documents/New folder/rating.csv")

# Buang rating -1
rating = rating[rating['rating'] != -1]

# Matriks user-anime
user_anime_matrix = rating.pivot_table(index='user_id', columns='anime_id', values='rating').fillna(0)

# ========================
# KONFIGURASI USER TETAP
# ========================
user_index = 0  # hanya menggunakan 1 user (User ID: 0)

# ========================
# BANGUN MODEL KNN
# ========================
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_anime_matrix)

# ========================
# STREAMLIT UI
# ========================
st.title("🎌 Sistem Rekomendasi Anime Berbasis KNN")
st.markdown("🔍 Rekomendasi berdasarkan tontonan Anda sebelumnya.")

# ========================
# PILIH GENRE FAVORIT
# ========================
if 'genre' in anime.columns:
    anime['genre'] = anime['genre'].fillna('').astype(str)
    anime['genre_list'] = anime['genre'].apply(lambda x: [g.strip() for g in x.split(',') if g])
    genre_unik = sorted(set(g for sublist in anime['genre_list'] for g in sublist))
else:
    genre_unik = []

selected_genre = st.selectbox("🎯 Pilih Genre Favorit:", ["Semua Genre"] + genre_unik)

# ========================
# PROSES REKOMENDASI
# ========================
user_vector = user_anime_matrix.loc[user_index].values.reshape(1, -1)
n_users = user_anime_matrix.shape[0]
n_neighbors = min(6, n_users - 1)

hasil_rekomendasi = []

if n_neighbors < 1:
    st.error("Tidak cukup data untuk mencari tetangga.")
else:
    distances, indices = model_knn.kneighbors(user_vector, n_neighbors=n_neighbors + 1)
    neighbors = [i for i in indices.flatten() if i != user_index]

    anime_scores = {}
    for neighbor in neighbors:
        neighbor_ratings = user_anime_matrix.iloc[neighbor]
        for anime_id, rating_val in neighbor_ratings.items():
            if rating_val > 0 and user_anime_matrix.loc[user_index, anime_id] == 0:
                anime_scores.setdefault(anime_id, []).append(rating_val)

    recommendations = {
        anime_id: np.mean(scores) for anime_id, scores in anime_scores.items()
    }

    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

    for anime_id, score in sorted_recommendations:
        row = anime[anime['anime_id'] == anime_id]
        if row.empty:
            continue
        nama = row['name'].values[0]
        genre = str(row['genre'].values[0]) if pd.notna(row['genre'].values[0]) else ''
        genre_list = [g.strip().lower() for g in genre.split(',') if g]

        if selected_genre == "Semua Genre" or selected_genre.lower() in genre_list:
            hasil_rekomendasi.append({
                'Anime': nama,
                'Genre': genre,
                'Skor Prediksi': round(score, 2)
            })

# ========================
# TAMPILKAN HASIL REKOMENDASI
# (DIPINDAHKAN KE ATAS)
# ========================
st.subheader("🎯 Rekomendasi Anime")

if not hasil_rekomendasi:
    st.warning("Tidak ada rekomendasi yang cocok dengan genre yang dipilih.")
else:
    hasil_df = pd.DataFrame(hasil_rekomendasi).head(10)
    st.dataframe(hasil_df)

    # Tombol Download CSV
    csv_buffer = StringIO()
    hasil_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Rekomendasi sebagai CSV",
        data=csv_buffer.getvalue(),
        file_name="rekomendasi_user0.csv",
        mime="text/csv"
    )

# ========================
# ANIME YANG SUDAH DITONTON
# (DIPINDAHKAN KE BAWAH)
# ========================
st.subheader("📺 Anime yang Sudah Ditonton")

user_ratings = rating[rating['user_id'] == user_index]
user_watched = pd.merge(user_ratings, anime, on='anime_id', how='inner')

expected_columns = ['anime_id', 'name', 'genre', 'rating']
available_columns = [col for col in expected_columns if col in user_watched.columns]
user_watched = user_watched[available_columns].rename(columns={
    'name': 'Anime',
    'genre': 'Genre',
    'rating': 'Rating'
})

if user_watched.empty:
    st.info("Anda belum menonton anime apapun.")
else:
    all_anime_watched = user_watched['Anime'].tolist()
    selected_anime = st.multiselect(
        "✅ Tampilkan hanya anime yang dipilih:",
        all_anime_watched,
        default=all_anime_watched
    )
    filtered_watched = user_watched[user_watched['Anime'].isin(selected_anime)]
    st.dataframe(filtered_watched.sort_values(by='Rating', ascending=False).reset_index(drop=True))
