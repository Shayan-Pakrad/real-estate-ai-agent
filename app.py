import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px

# تنظیمات ظاهری
st.set_page_config(page_title="پلتفرم پیش‌بینی خودکار", layout="wide")

# استایل RTL برای فارسی‌سازی
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn&display=swap');
    html, body, [class*="css"] { font-family: 'Vazirmatn', sans-serif; direction: rtl; text-align: right; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 پلتفرم هوشمند آموزش و پیش‌بینی")
st.write("فایل CSV خود را آپلود کنید تا مدل اختصاصی شما در لحظه ساخته شود.")

# --- مرحله ۱: آپلود داده‌ها ---
uploaded_file = st.file_uploader("فایل داده‌های خود را انتخاب کنید (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ فایل با موفقیت بارگذاری شد.")
    
    with st.expander("👀 مشاهده ۵ سطر اول داده‌ها"):
        st.dataframe(df.head(), use_container_width=True)

    # --- مرحله ۲: تنظیمات آموزش ---
    st.sidebar.header("⚙️ تنظیمات مدل")
    
    # انتخاب ستون هدف (Target)
    all_columns = df.columns.tolist()
    target_col = st.sidebar.selectbox("کدام ستون را پیش‌بینی کنیم؟ (Target)", all_columns, index=len(all_columns)-1)
    
    # شناسایی خودکار ویژگی‌ها (ستون‌های عددی به جز هدف)
    features_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != target_col]
    
    selected_features = st.sidebar.multiselect("ویژگی‌های ورودی را تایید کنید:", features_cols, default=features_cols)

    if st.sidebar.button("🚀 شروع آموزش مدل"):
        with st.spinner('در حال پردازش داده‌ها و آموزش مدل...'):
            # آماده‌سازی داده‌ها
            X = df[selected_features].dropna()
            y = df.loc[X.index, target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # پیش‌پردازش
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # آموزش مدل (Random Forest بر اساس نوت‌بوک شما)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # ارزیابی
            preds = model.predict(X_test_scaled)
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            
            # ذخیره در session_state برای استفاده در بخش پیش‌بینی
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['features'] = selected_features
            st.session_state['trained'] = True
            
            st.balloons()
            
            # نمایش نتایج
            c1, c2 = st.columns(2)
            c1.metric("دقت مدل (R2 Score)", f"{r2:.2%}")
            c2.metric("میانگین خطا (MAE)", f"{mae:.2f}")

    # --- مرحله ۳: بخش پیش‌بینی ---
    if st.session_state.get('trained'):
        st.divider()
        st.header("🔮 پیش‌بینی مقدار جدید")
        
        with st.form("prediction_form"):
            st.write("مقادیر ورودی را وارد کنید:")
            input_values = []
            cols = st.columns(3)
            
            for i, feat in enumerate(st.session_state['features']):
                with cols[i % 3]:
                    val = st.number_input(f"{feat}", value=float(df[feat].mean()))
                    input_values.append(val)
            
            submit = st.form_submit_button("محاسبه پیش‌بینی")
            
            if submit:
                model = st.session_state['model']
                scaler = st.session_state['scaler']
                
                input_array = np.array([input_values])
                input_scaled = scaler.transform(input_array)
                prediction = model.predict(input_scaled)
                
                st.write(f"### 🎯 مقدار پیش‌بینی شده برای {target_col}:")
                st.subheader(f"{prediction[0]:,.2f}")
                
                # نمایش نمودار اهمیت ویژگی‌ها
                st.write("---")
                st.write("📊 اهمیت هر ویژگی در این پیش‌بینی:")
                importance_df = pd.DataFrame({
                    'ویژگی': st.session_state['features'],
                    'اهمیت': model.feature_importances_
                }).sort_values(by='اهمیت', ascending=False)
                
                fig = px.bar(importance_df, x='اهمیت', y='ویژگی', orientation='h', color='اهمیت')
                st.plotly_chart(fig, use_container_width=True)

else:
    # صفحه خوش‌آمدگویی در صورتی که فایلی آپلود نشده باشد
    st.info("👈 برای شروع، لطفاً یک فایل CSV از منوی سمت راست آپلود کنید.")
    st.image("https://img.freepik.com/free-vector/data-analysis-concept-illustration_114360-1611.jpg", width=400)
