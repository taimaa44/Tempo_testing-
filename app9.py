import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import folium
from streamlit_folium import st_folium

# ==============================
# 1. قراءة البيانات
# ==============================
ds = xr.open_dataset(r"C:\\Users\\User\\OneDrive\\Desktop\\tempotraining\\TEMPO_NO2_L3_V03_20250916T210309Z_S012(1).nc")
df = ds["weight"].to_dataframe().reset_index()
df = df.dropna(subset=["weight"])
df["weight"] = np.where(df["weight"] < 0, np.nan, df["weight"])
df = df.dropna(subset=["weight"])

# أخذ عينة صغيرة لتسريع العرض
sampled_df = df.sample(frac=0.02, random_state=42)

# ==============================
# 2. تدريب نموذج Gradient Boosting
# ==============================
X = sampled_df[["latitude", "longitude"]]
y = sampled_df["weight"]
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
gbr.fit(X, y)
sampled_df["weight_pred"] = gbr.predict(X)

# ==============================
# 3. واجهة Streamlit
# ==============================
st.title("🌍 خريطة تفاعلية لتوقع تلوث NO₂")

# إنشاء خريطة Folium
m = folium.Map(location=[sampled_df["latitude"].mean(), sampled_df["longitude"].mean()], zoom_start=5, tiles="CartoDB positron")

# ==============================
# 4. وضع CircleMarkers للمناطق عالية التلوث فقط
# ==============================
threshold = sampled_df["weight_pred"].quantile(0.95)  # أعلى 5%
high_pollution = sampled_df[sampled_df["weight_pred"] >= threshold]

for _, row in high_pollution.iterrows():
    popup_html = f"""
    <b>مستوى NO₂: {row['weight_pred']:.2f}</b><br>
    ⚠ منطقة ملوثة بشدة!<br>
    <button onclick="alert('تم إرسال اقتراح لزراعة الأشجار 🌳')">🌳 زراعة الأشجار</button>
    """
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=8,
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.7,
        popup=folium.Popup(popup_html, max_width=300)
    ).add_to(m)

# ==============================
# 5. عرض الخريطة داخل Streamlit
# ==============================
st_folium(m, width=900, height=600)
