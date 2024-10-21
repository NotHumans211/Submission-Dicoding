import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the app
st.title('Bike-Sharing Analysis App')

# Load dataset
st.sidebar.title("Upload Dataset")
day_data = st.sidebar.file_uploader("Upload day.csv", type=["csv"])
hour_data = st.sidebar.file_uploader("Upload hour.csv", type=["csv"])

if day_data is not None:
    day_data = pd.read_csv(day_data)
    st.write("## Day Dataset")
    st.write(day_data.head())

    # Visualization 1: Pengaruh Cuaca dan Musim terhadap Penyewaan Sepeda
    st.subheader("Pengaruh Cuaca dan Musim terhadap Penyewaan Sepeda")
    season_weather_group = day_data.groupby(['season', 'weathersit']).agg({'cnt': 'mean'}).reset_index()
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='season', y='cnt', hue='weathersit', data=season_weather_group, palette='coolwarm', ax=ax1)
    ax1.set_title('Pengaruh Cuaca dan Musim terhadap Penyewaan Sepeda')
    ax1.set_xlabel('Musim')
    ax1.set_ylabel('Rata-rata Penyewaan Sepeda')
    st.pyplot(fig1)

    # Visualization 2: Tren Penyewaan Sepeda sepanjang Tahun
    st.subheader("Tren Penyewaan Sepeda sepanjang Tahun")
    monthly_trend = day_data.groupby('mnth').agg({'cnt': 'mean'}).reset_index()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='mnth', y='cnt', data=monthly_trend, marker='o', ax=ax2)
    ax2.set_title('Tren Penyewaan Sepeda sepanjang Tahun')
    ax2.set_xlabel('Bulan')
    ax2.set_ylabel('Rata-rata Penyewaan Sepeda')
    st.pyplot(fig2)

    # Visualization 3: Hubungan antara Suhu dan Penyewaan Sepeda
    st.subheader("Hubungan antara Suhu dan Penyewaan Sepeda")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='temp', y='cnt', data=day_data, alpha=0.5, ax=ax3)
    ax3.set_title('Hubungan antara Suhu dan Penyewaan Sepeda')
    ax3.set_xlabel('Suhu (Normalized)')
    ax3.set_ylabel('Jumlah Penyewaan Sepeda')
    st.pyplot(fig3)

    # Visualization 4: Distribusi Pengguna Kasual vs Terdaftar
    st.subheader("Distribusi Pengguna Kasual vs Terdaftar")
    user_type_counts = day_data[['casual', 'registered']].sum()
    fig4, ax4 = plt.subplots(figsize=(8, 8))
    ax4.pie(user_type_counts, labels=['Casual', 'Registered'], autopct='%1.1f%%', startangle=90)
    ax4.set_title('Distribusi Pengguna Kasual vs Terdaftar')
    st.pyplot(fig4)

    # Heatmap: Korelasi antara Variabel-variabel
    st.subheader("Korelasi antara Variabel-variabel")
    corr = day_data.corr()
    fig5, ax5 = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax5)
    ax5.set_title('Heatmap Korelasi antara Variabel-variabel')
    st.pyplot(fig5)

# Instruction to run the app
st.sidebar.info("Run this app with the following command in your terminal: `streamlit run bike_sharing_app.py`")
