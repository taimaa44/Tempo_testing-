import xarray as xr
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# افتح الملف
ds = xr.open_dataset(r"C:\\Users\User\\OneDrive\Desktop\\tempotraining\\TEMPO_NO2_L3_V03_20250916T210309Z_S012(1).nc")
print(ds["time"].values)

# اعرض الملخص عن المتغيرات
#print(ds)
#print(ds["weight"].shape)  
# استعراض أسماء الفاريبلز
#print(ds.data_vars)
#print(ds["weight"])           # معلومات عن المتغير
#print(ds["weight"].values)    # المصفوفة نفسها (قيم NO2)
#print(ds["weight"].isel(latitude=1000, longitude=2000).values)

import numpy as np

no2 = ds["weight"].values

print(ds.data_vars)

# استبدل القيم السالبة بـ NaN
no2 = np.where(no2 < 0, np.nan, no2)
#print(no2)
# اختياري: استبدال NaN بمتوسط/ميديان
# from numpy import nanmean
# no2 = np.where(np.isnan(no2), np.nanmean(no2), no2)

df = ds["weight"].to_dataframe().reset_index()
#print(df.head())

#print("the number of data ", len(df))
df_clean = df.dropna(subset=["weight"])
print(df_clean.head())

#print("the number of data ", len(df))
#print("mean:", df["weight"].mean())
#print("max value", df["weight"].max())


subset = df_clean[
    (df_clean["latitude"].between(30, 40)) &   # خط العرض
    (df_clean["longitude"].between(-100, -90)) # خط الطول
]
print(subset)
sampled_df = df_clean.sample(frac=0.01, random_state=42)  # 1% من البيانات
print(len(sampled_df))

import pandas as pd
from sklearn.model_selection import train_test_split

# افترض أن df_small هو الـ subset الصغير
df = sampled_df.copy()  # df_small = subset / sample

# Features و Target
X = df[["latitude", "longitude"]]  # فقط الإحداثيات
y = df["weight"]

# تقسيم البيانات train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR": SVR()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "R2": r2}
    
# عرض النتائج
results_df = pd.DataFrame(results).T
print(results_df.sort_values("R2", ascending=False))


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

# افترض أن df_small هو subset نظيف
df = sampled_df.copy()  

# Features و Target
X = df[["latitude", "longitude"]]
y = df["weight"]

# تدريب GradientBoosting على كامل الـ subset
gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gbr.fit(X, y)

# Prediction على نفس الـ subset (أو على شبكة نقاط جديدة)
df["weight_pred"] = gbr.predict(X)

# رسم الخريطة
plt.figure(figsize=(10, 6))
sc = plt.scatter(
    df["longitude"], df["latitude"], 
    c=df["weight_pred"], cmap="inferno", s=10
)
plt.colorbar(sc, label="Predicted NO₂")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("GradientBoosting Predicted NO₂ Concentration")
plt.show()
