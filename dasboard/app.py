import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest


# Load dataset
@st.cache
def load_data():
    day_data = pd.read_csv('data/day.csv')
    return day_data

data = load_data()

# Display dataset
st.title("Analisis Data Penyewaan Sepeda")
st.write("Dataset Penyewaan Sepeda:")
st.dataframe(data.head())

# Explore Distribution of Bike Rentals
st.subheader("Trend Penyewaan Sepeda Harian")
plt.figure(figsize=(10, 6))
plt.plot(data['dteday'], data['cnt'])
plt.title('Trend Penyewaan Sepeda Harian')
plt.xlabel('Tanggal')
plt.ylabel('Jumlah Penyewaan')
plt.grid(True)
st.pyplot(plt)

# Insight
st.write("""
Terlihat ada tren musiman dalam penyewaan sepeda, dengan peningkatan selama musim panas dan penurunan di musim dingin.
""")

# Pertanyaan 1: Pengaruh Cuaca dan Musim terhadap Penyewaan Sepeda
st.subheader("Pengaruh Cuaca dan Musim terhadap Penyewaan Sepeda")
season_weather_group = data.groupby(['season', 'weathersit'])['cnt'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='season', y='cnt', hue='weathersit', data=season_weather_group, palette='viridis')
plt.title('Pengaruh Cuaca dan Musim terhadap Penyewaan Sepeda', fontsize=16)
plt.xlabel('Musim', fontsize=12)
plt.ylabel('Rata-rata Jumlah Penyewaan', fontsize=12)
plt.xticks(ticks=[0, 1, 2, 3], labels=['Spring', 'Summer', 'Fall', 'Winter'], fontsize=10)
plt.legend(title='Kondisi Cuaca', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
st.pyplot(plt)

# Insight
st.write("""
Cuaca Fall dan Summer memiliki tingkat penyewaan yang tinggi.
""")

# Pertanyaan 2: Perbedaan Pola Penyewaan Kasual vs Terdaftar
st.subheader("Perbedaan Pola Penyewaan Kasual vs Terdaftar")
user_type_group = data.groupby(['workingday']).agg({'casual': 'mean', 'registered': 'mean'}).reset_index()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Casual rentals pie chart
ax1.pie(user_type_group['casual'], labels=['Non-Working Day', 'Working Day'],
        autopct='%1.1f%%', startangle=90, colors=['blue', 'lightblue'],
        explode=[0.1, 0])
ax1.set_title('Casual Rentals on Working vs. Non-Working Days')

# Registered rentals pie chart
ax2.pie(user_type_group['registered'], labels=['Non-Working Day', 'Working Day'],
        autopct='%1.1f%%', startangle=90, colors=['green', 'lightgreen'],
        explode=[0.1, 0])
ax2.set_title('Registered Rentals on Working vs. Non-Working Days')

st.pyplot(fig)

# Pertanyaan 3: Pengaruh Kecepatan Angin dan Kelembaban
st.subheader("Pengaruh Kecepatan Angin dan Kelembaban terhadap Penyewaan Sepeda")
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(x='windspeed', y='cnt', hue='hum', data=data, palette='coolwarm', size='hum')
plt.title('Pengaruh Kecepatan Angin dan Kelembaban terhadap Penyewaan Sepeda')
plt.xlabel('Kecepatan Angin')
plt.ylabel('Jumlah Penyewaan Sepeda')
plt.colorbar(scatter.collections[0], label='Kelembaban')
st.pyplot(plt)

# Pertanyaan 4: Pola Penyewaan Berdasarkan Musim
st.subheader("Pola Penyewaan Berdasarkan Musim antara Pengguna Kasual dan Terdaftar")
season_user_type_group = data.groupby(['season'])[['casual', 'registered']].mean().reset_index()

# Line plot for casual and registered users
plt.figure(figsize=(12, 6))
sns.lineplot(x='season', y='casual', data=season_user_type_group, label='Casual', marker='o')
sns.lineplot(x='season', y='registered', data=season_user_type_group, label='Registered', marker='o')
plt.title('Pola Penyewaan Sepeda Berdasarkan Musim antara Pengguna Kasual dan Terdaftar')
plt.xlabel('Musim')
plt.ylabel('Rata-rata Jumlah Penyewaan')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Spring', 'Summer', 'Fall', 'Winter'])
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Bar plot for casual and registered users
plt.figure(figsize=(12, 6))
sns.barplot(x='season', y='casual', data=season_user_type_group, color='blue', label='Casual')
sns.barplot(x='season', y='registered', data=season_user_type_group, color='green', label='Registered', alpha=0.5)
plt.title('Pola Penyewaan Sepeda Berdasarkan Musim antara Pengguna Kasual dan Terdaftar')
plt.xlabel('Musim')
plt.ylabel('Rata-rata Jumlah Penyewaan')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Spring', 'Summer', 'Fall', 'Winter'])
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Pertanyaan 5: Pengaruh Hari Libur
st.subheader("Pengaruh Hari Libur terhadap Penyewaan Sepeda")
holiday_group = data.groupby(['holiday']).agg({'cnt': 'mean'}).reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(x='holiday', y='cnt', data=holiday_group, palette='viridis')
plt.title('Rata-Rata Penyewaan Sepeda pada Hari Libur vs Hari Kerja')
plt.xlabel('Hari Libur (1 = Ya, 0 = Tidak)')
plt.ylabel('Rata-Rata Jumlah Penyewaan')
plt.xticks(ticks=[0, 1], labels=['Hari Kerja', 'Hari Libur'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(plt)

# Analisis Lanjutan: Model Prediksi Menggunakan Random Forest
st.subheader("Model Prediksi Menggunakan Random Forest")
features = ['temp', 'hum', 'windspeed', 'season', 'weathersit', 'workingday', 'holiday', 'weekday', 'mnth', 'yr']
X = data[features]
y = data['cnt']

# Scaling features
scaler = StandardScaler()
X[['temp', 'hum', 'windspeed']] = scaler.fit_transform(X[['temp', 'hum', 'windspeed']])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"R² Score: {r2}")

# Feature importance
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
st.subheader("Feature Importance")
fig, ax = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
st.pyplot(fig)

# Clustering Pengguna Berdasarkan Pola Penyewaan Menggunakan K-Means
st.subheader("Clustering Pengguna Berdasarkan Pola Penyewaan")
cluster_features = ['casual', 'registered', 'temp', 'hum', 'windspeed']
scaler = StandardScaler()
cluster_data = scaler.fit_transform(data[cluster_features])

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(cluster_data)

# Visualisasi hasil clustering
plt.figure(figsize=(10, 6))
sns.scatterplot(x='temp', y='cnt', hue='cluster', data=data, palette='viridis', s=100)
plt.title('Clustering Pengguna Berdasarkan Pola Penyewaan Sepeda')
plt.xlabel('Suhu')
plt.ylabel('Jumlah Penyewaan Sepeda')
plt.legend(title='Cluster')
plt.grid(True)
st.pyplot(plt)

# Analisis Pengaruh Event Terhadap Penyewaan Sepeda (Anomaly Detection)
st.subheader("Analisis Pengaruh Event Terhadap Penyewaan Sepeda (Deteksi Anomali)")
anomaly_features = ['temp', 'hum', 'windspeed', 'season', 'cnt']

# Terapkan Isolation Forest untuk mendeteksi anomali
isolation_forest = IsolationForest(contamination=0.05, random_state=42)
data['anomaly'] = isolation_forest.fit_predict(data[anomaly_features])

# Visualisasi anomali
plt.figure(figsize=(10, 6))
sns.scatterplot(x='temp', y='cnt', hue='anomaly', data=data, palette={1: 'blue', -1: 'red'}, s=100)
plt.title('Deteksi Anomali Penyewaan Sepeda Berdasarkan Suhu')
plt.xlabel('Suhu')
plt.ylabel('Jumlah Penyewaan Sepeda')
plt.legend(title='Anomali', loc='upper right', labels=['Normal', 'Anomali'])
plt.grid(True)
st.pyplot(plt)

# Tuning Hyperparameter untuk Meningkatkan Performa Model
st.subheader("Tuning Hyperparameter untuk Meningkatkan Performa Model")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Buat objek GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Lakukan pencarian hyperparameter terbaik
grid_search.fit(X_train, y_train)

# Cetak hyperparameter terbaik
st.write("Best parameters:", grid_search.best_params_)

# Gunakan model dengan hyperparameter terbaik untuk prediksi
best_rf_model = grid_search.best_estimator_
y_pred_best = best_rf_model.predict(X_test)

# Evaluasi model dengan hyperparameter terbaik
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

st.write("Mean Squared Error (MSE):", mse_best)
st.write("R² Score:", r2_best)

# Membandingkan dengan model regresi lainnya
st.subheader("Membandingkan dengan Model Regresi Lainnya")
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# 1. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# 2. Decision Tree Regression
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

# 3. Support Vector Regression (SVR)
svr_model = SVR()
svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

# Print results for comparison
st.write("\nModel Comparison:")
st.write("-" * 30)
st.write("Linear Regression:")
st.write("MSE:", mse_lr)
st.write("R²:", r2_lr)

st.write("\nDecision Tree Regression:")
st.write("MSE:", mse_dt)
st.write("R²:", r2_dt)

st.write("\nSupport Vector Regression (SVR):")
st.write("MSE:", mse_svr)
st.write("R²:", r2_svr)

st.write("\nRandom Forest Regression (from previous code):")
st.write("MSE:", mse)
st.write("R²:", r2)

# Conclusion
st.subheader("Kesimpulan")
st.write("""
Dari analisis data di atas, dapat disimpulkan bahwa:
1. **Pengaruh Cuaca dan Musim:**
   - Cuaca cerah dan musim panas memiliki korelasi positif yang kuat terhadap peningkatan jumlah penyewaan sepeda. Selama musim panas, penyewaan sepeda mencapai puncaknya, terutama pada hari-hari dengan kondisi cuaca cerah.
   - Sebaliknya, pada musim dingin dan saat cuaca buruk (hujan atau salju), jumlah penyewaan sepeda mengalami penurunan signifikan.

2. **Perilaku Pengguna Kasual vs Terdaftar:**
   - Pengguna kasual cenderung lebih aktif pada akhir pekan dan hari libur, menunjukkan pola penggunaan yang lebih bersifat rekreasi.
   - Pengguna terdaftar memiliki pola penggunaan yang lebih stabil sepanjang minggu, dengan lebih banyak penyewaan pada hari kerja, menunjukkan bahwa mereka mungkin lebih sering menggunakan sepeda untuk kebutuhan transportasi harian.

3. **Pengaruh Suhu dan Kelembaban:**
   - Terdapat hubungan positif antara suhu dan jumlah penyewaan sepeda. Saat suhu meningkat, penyewaan juga meningkat hingga suhu tertentu. Namun, pada suhu ekstrem yang terlalu panas, penyewaan sepeda cenderung menurun.
   - Kelembaban juga mempengaruhi penyewaan, tetapi dampaknya tidak sebesar suhu. Kondisi cuaca yang terlalu lembap dapat sedikit mengurangi jumlah pengguna yang menyewa sepeda.

4. **Distribusi Pengguna Kasual dan Terdaftar:**
   - Dari total jumlah penyewaan sepeda, pengguna terdaftar mendominasi dibandingkan pengguna kasual. Hal ini menunjukkan bahwa sebagian besar penyewaan berasal dari pengguna reguler yang mungkin menggunakan sepeda untuk keperluan sehari-hari seperti bekerja atau bersekolah.

5. **Korelasi antara Variabel:**
   - Beberapa variabel seperti suhu dan kecepatan angin menunjukkan korelasi yang signifikan terhadap jumlah penyewaan sepeda. Suhu memiliki korelasi positif yang kuat, sementara kecepatan angin menunjukkan korelasi negatif ringan, di mana angin yang lebih kencang sedikit mengurangi jumlah penyewaan.

6. **Tren Penyewaan Sepanjang Tahun:**
   - Tren penyewaan sepeda cenderung meningkat selama bulan-bulan musim semi hingga musim panas (April hingga September). Pada bulan-bulan musim gugur dan musim dingin (Oktober hingga Februari), jumlah penyewaan menurun, yang mungkin disebabkan oleh kondisi cuaca yang kurang mendukung.
""")

# Run the Streamlit app
if __name__ == "__main__":
   st.run()