import numpy as np
import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# -------------------------
st.set_page_config(page_title="E-Commerce Price Prediction", layout="wide")

# -------------------------
# LOAD CSS SAFELY
# -------------------------
def load_css():
    css_path = "assets/style.css"
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -------------------------
# LOAD DATA, MODEL & ENCODER
# -------------------------
df = pd.read_csv("amazon.csv")
model = joblib.load("price_model.pkl")
le = joblib.load("label_encoder.pkl")
metrics = joblib.load("model_metrics.pkl")

# -------------------------
# CLEAN CATEGORY
# -------------------------
df["category"] = df["category"].astype(str).str.split("|").str[0]
df["category"] = df["category"].str.replace("&", " & ").str.strip()

# -------------------------
# CLEAN NUMERIC COLUMNS
# -------------------------
df["actual_price"] = (
    df["actual_price"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
)

df["rating_count"] = (
    df["rating_count"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df["discount_percentage"] = (
    df["discount_percentage"]
    .astype(str)
    .str.replace("%", "", regex=False)
)

df["actual_price"] = pd.to_numeric(df["actual_price"], errors="coerce")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")
df["discount_percentage"] = pd.to_numeric(df["discount_percentage"], errors="coerce")

df = df.dropna(subset=["category", "rating", "rating_count", "discount_percentage", "actual_price"])

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("", ["Home", "Dataset", "EDA", "Modeling", "Results", "About"])

# =========================
# HOME PAGE
# =========================
if page == "Home":

    st.title("🛒 E-Commerce Price Prediction Web App")
    st.write("Select a product to predict its price using Machine Learning.")

    col1, col2 = st.columns([2, 1])

    with col1:

        st.markdown("### 📦 Select Product")

        product_name = st.selectbox(
            "",
            sorted(df["product_name"].unique()),
            label_visibility="collapsed"
        )

        selected = df[df["product_name"] == product_name].iloc[0]

        category = selected["category"]
        rating = selected["rating"]
        rating_count = selected["rating_count"]
        discount_percentage = selected["discount_percentage"]
        actual_price = selected["actual_price"]

        st.markdown("### 📋 Product Details")
        st.write(f"**Product Name:** {product_name}")
        st.write(f"**Category:** {category}")
        st.write(f"⭐ Rating: {rating}")
        st.write(f"🗳 Reviews: {int(rating_count):,}")
        st.write(f"🏷 Discount: {discount_percentage}%")

        # 🔥 Predict Button
        if st.button("💰 Predict Price"):

            # Safe encoding
            if category in le.classes_:
                category_encoded = le.transform([category])[0]
            else:
                st.error("Category not found in encoder.")
                st.stop()

            input_data = pd.DataFrame(
                [[category_encoded, rating, rating_count, discount_percentage]],
                columns=[
                    "category_encoded",
                    "rating",
                    "rating_count",
                    "discount_percentage"
                ]
            )

            # Convert log prediction to actual value
            prediction_log = model.predict(input_data)[0]
            prediction = np.expm1(prediction_log)

            difference = prediction - actual_price

            st.markdown("## 🔎 Prediction Result")

            colA, colB = st.columns(2)

            with colA:
                st.success(f"Predicted Price: ₹ {int(prediction):,}")

            with colB:
                st.info(f"Actual Price: ₹ {int(actual_price):,}")

            st.markdown("---")

            if difference > 0:
                st.warning(f"Model Overestimated by ₹ {int(abs(difference)):,}")
            elif difference < 0:
                st.error(f"Model Underestimated by ₹ {int(abs(difference)):,}")
            else:
                st.success("Perfect Prediction!")

    with col2:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/1170/1170678.png",
            width=250
        )

# =========================
# OTHER PAGES
# =========================
elif page == "Dataset":
    st.title("📊 Dataset Preview")
    st.dataframe(df.head())

elif page == "EDA":

    st.title("📈 Exploratory Data Analysis")

    st.subheader("Category Distribution")
    st.bar_chart(df["category"].value_counts())

    st.subheader("Rating Distribution")
    st.bar_chart(df["rating"].value_counts().sort_index())

    st.subheader("Discount vs Rating")
    st.scatter_chart(
        df,
        x="discount_percentage",
        y="rating"
    )

    st.subheader("Top 10 Most Expensive Products")
    top_expensive = df.sort_values("actual_price", ascending=False).head(10)
    st.dataframe(
        top_expensive[["product_name", "category", "actual_price"]],
        use_container_width=True
    )

elif page == "Modeling":
    st.title("🤖 Model Information")
    st.write("Model Used: Random Forest Regressor")
    st.write("Features Used:")
    st.write("- Category")
    st.write("- Rating")
    st.write("- Rating Count")
    st.write("- Discount Percentage")

elif page == "Results":

    st.title("📊 Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", round(metrics["mae"], 2))
    col2.metric("RMSE", round(metrics["rmse"], 2))
    col3.metric("R² Score", round(metrics["r2"], 3))

    st.markdown("### Interpretation")

    st.write("""
    - MAE shows average prediction error.
    - RMSE penalizes large errors more.
    - R² score shows how well model explains variance.
    """)

elif page == "About":
    st.title("ℹ️ About Project")
    st.write("E-Commerce Price Prediction Web App")
    st.write("Built using Python, Streamlit, and Machine Learning.")