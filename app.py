import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="GES Elektrik Üretim Tahmini", layout="centered")
st.title("☀️ Güneş Enerjisi Santrali (GES) Güç Tahmin Sistemi")
st.markdown("Hava durumu verilerini girerek üretilecek elektrik miktarını (kW) tahmin edin.")

@st.cache_resource
def load_models():
    # Modeli, özellikleri VE SCALER'ı yüklüyoruz
    model = joblib.load('knn_sampiyon_model.pkl')
    features = joblib.load('kullanilan_ozellikler.pkl')
    scaler = joblib.load('minmax_scaler.pkl')
    return model, features, scaler

model, selected_features, scaler = load_models()

feature_mapping = {
    'shortwave_radiation_backwards_sfc': 'Kısa Dalga Güneş Radyasyonu',
    'angle_of_incidence': 'Güneş Işığının Geliş Açısı',
    'total_cloud_cover_sfc': 'Toplam Bulutluluk Oranı (0-100)',
    'azimut': 'Azimut Açısı (Güneş Yönü)',
    'zenit': 'Zenit Açısı (Tepe Açısı)'
}

st.subheader("Hava Durumu Parametrelerini Girin")
user_inputs = {}

# Sadece seçilen özelliklerin giriş kutularını oluştur
for feature in selected_features:
    display_name = feature_mapping.get(feature, feature)
    user_inputs[feature] = st.number_input(f"{display_name} değerini girin:", value=0.0)

if st.button("Üretimi Tahmin Et"):
    try:
        # Kullanıcının girdiği verileri DataFrame'e çevir
        input_data = pd.DataFrame([user_inputs])
        
        # KRİTİK NOKTA: Girilen ham veriyi 0-1 arasına scale et (Eğitimdeki gibi)
        input_scaled = scaler.transform(input_data)
        
        # Tahmini yap (Model eksi bir değer üretmesin diye max(0) koyduk)
        tahmin = model.predict(input_scaled)[0]
        tahmin_sonuc = max(0, tahmin) 
        
        st.success(f"Beklenen Elektrik Üretimi: {tahmin_sonuc:.2f} kW")
    except Exception as e:
        st.error(f"Hata oluştu: {e}")
