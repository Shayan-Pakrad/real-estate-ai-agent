import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# تنظیمات صفحه
st.set_page_config(page_title="پیش‌بینی قیمت مسکن صادقیه", layout="centered")

# عنوان برنامه
st.title("🏠 سامانه پیش‌بینی قیمت ملک (صادقیه)")
st.write("این برنامه بر اساس داده‌های واقعی، قیمت تقریبی خانه را پیش‌بینی می‌کند.")

# بارگذاری داده‌ها برای آموزش مدل
@st.cache_resource
def train_model():
    # خواندن فایل داده‌ها
    df = pd.read_csv('data_sadeghiyeh.csv')
    
    # پیش‌پردازش طبق نوت‌بوک شما
    df = df.dropna()
    df = df[(df["Price"] >= 5e8) & (df["Price"] <= 5e10)] # فیلتر قیمت‌های پرت
    
    # مهندسی ویژگی‌ها
    df["building_age"] = 1404 - df["Year Of Construction"]
    df["log_price"] = np.log1p(df["Price"])
    
    # تعریف ویژگی‌ها (X) و هدف (y)
    X = df[["Area", "Room", "Floor Number", "Parking", "Warehouse", "Elevator", "building_age"]]
    y = df["log_price"]
    
    # ساخت خط لوله پردازش (Pipeline)
    num_cols = ["Area", "Room", "Floor Number", "building_age"]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols)
    ], remainder='passthrough')
    
    model = Pipeline([
        ("prep", preprocessor),
        ("model", SVR(C=1.0, epsilon=0.1))
    ])
    
    model.fit(X, y)
    return model

# اجرای آموزش مدل
try:
    model = train_model()
except Exception as e:
    st.error(f"خطا در بارگذاری داده‌ها: {e}")
    st.stop()

# بخش ورودی‌های کاربر
st.subheader("📋 مشخصات ملک را وارد کنید:")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("متراژ (متر مربع)", min_value=10, max_value=500, value=100)
    rooms = st.selectbox("تعداد اتاق", [0, 1, 2, 3, 4, 5], index=2)
    year = st.number_input("سال ساخت (شمسی)", min_value=1350, max_value=1404, value=1390)

with col2:
    floor = st.number_input("طبقه", min_value=0, max_value=30, value=2)
    parking = st.checkbox("پارکینگ دارد", value=True)
    warehouse = st.checkbox("انباری دارد", value=True)
    elevator = st.checkbox("آسانسور دارد", value=True)

# محاسبه سن ساختمان برای مدل
age = 1404 - year

# دکمه پیش‌بینی
if st.button("💰 محاسبه قیمت"):
    # آماده‌سازی داده برای پیش‌بینی
    input_data = pd.DataFrame([[area, rooms, floor, parking, warehouse, elevator, age]], 
                             columns=["Area", "Room", "Floor Number", "Parking", "Warehouse", "Elevator", "building_age"])
    
    # انجام پیش‌بینی
    prediction_log = model.predict(input_data)
    prediction_actual = np.expm1(prediction_log)[0]
    
    # نمایش نتیجه
    st.success(f"قیمت تخمینی: {prediction_actual:,.0f} تومان")
    st.info(f"قیمت هر متر مربع: {prediction_actual/area:,.0f} تومان")
