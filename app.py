import io
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Property Intelligence Dashboard",
    page_icon="🏠",
    layout="wide"
)

# =========================
# LOAD FILES
# =========================

artifacts = joblib.load(
    "Real_Estate_Price_Predictor.pkl"
)

model = artifacts["model"]
std = artifacts["std"]
ohe = artifacts["ohe"]
te = artifacts["te"]

csv_path = Path("ML_cleaned_real_estate.csv")
raw_text = csv_path.read_text(encoding="utf-8-sig")
clean_lines = []
for line in raw_text.splitlines():
    line = line.strip()
    if not line:
        continue
    if line.startswith('"') and line.endswith('"'):
        line = line[1:-1]
    clean_lines.append(line)

csv_data = "\n".join(clean_lines)
df = pd.read_csv(io.StringIO(csv_data), sep=",")

df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.strip('"')
    .str.lstrip("\ufeff")
)

numeric_cols = [
    "Bathroom",
    "Balcony",
    "BHK",
    "Price_clean",
    "Area(sqft)",
    "Floor_No",
    "Total_Floors",
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# TITLE
# =========================

st.markdown("""
<div style="
background:linear-gradient(135deg,#1f4e79,#2f80ed);
padding:25px;
border-radius:20px;
text-align:center;
margin-bottom:20px;
">

<h1 style="
color:white;
font-size:42px;
margin-bottom:5px;
">
🏠 Smart Property Valuation System
</h1>

<p style="
color:white;
font-size:18px;
">
Estimate Property Prices Across Major Indian Cities Using AI
</p>

</div>
""", unsafe_allow_html=True)

st.markdown("---")

# KPI CARDS

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.info("""
🏙 Cities

# 5
""")

with c2:
    st.info("""
🏠 Properties

# 2877
""")

with c3:
    st.info("""
📊 Features

# 12
""")

with c4:
    st.info("""
🤖 Model

# RF
""")
st.markdown('------')

# INFO BOX

st.markdown("""
<div style="
background:#14304a;
padding:25px;
border-radius:15px;
margin-top:25px;
margin-bottom:25px;
">

<h4 style="color:#4da6ff;">
🏡 Smart Property Valuation
</h4>

<p>
Get an estimated market value for your property based on:
</p>

<ul>
<li>📍 Property Location</li>
<li>🏠 Property Size & Layout</li>
<li>🏢 Floor Information</li>
<li>🛋️ Furnishing Status</li>
<li>🌇 Facing Direction</li>
<li>🏙️ City & Area Trends</li>
</ul>

</div>
""", unsafe_allow_html=True)



# =========================
# INPUTS
# =========================

st.subheader("🏡 Property Specifications")

col1, col2 = st.columns(2)

with col1:
    bathroom = st.number_input("Bathroom",1.0,step=1.0)
    bhk = st.number_input("BHK",1.0,step=1.0)
    floor_no = st.number_input("Floor Number",0.0,step=1.0)

with col2:
    balcony = st.number_input("Balcony",0.0,step=1.0)
    area = st.number_input("Area (sqft)",100.0,step=50.0)
    total_floors = st.number_input("Total Floors",1.0,step=1.0)

status = st.selectbox(
    "Status",
    df["Status"].unique()
)

transaction = st.selectbox(
    "Transaction",
    df["Transaction"].unique()
)

furnishing = st.selectbox(
    "Furnishing",
    df["Furnishing"].unique()
)

facing = st.selectbox(
    "Facing",
    df["Facing"].unique()
)

city = st.selectbox(
    "City",
    df["City"].unique()
)

filtered_areas = sorted(
    df[df["City"] == city]["Area_Name"].dropna().unique()
)

area_name = st.selectbox(
    "Area Name",
    filtered_areas
)

# =========================
# PREDICTION
# =========================

if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "Status":[status],
        "Transaction":[transaction],
        "Furnishing":[furnishing],
        "Facing":[facing],
        "City":[city],
        "Area_Name":[area_name],

        "Bathroom":[bathroom],
        "Balcony":[balcony],
        "BHK":[bhk],
        "Area(sqft)":[area],
        "Floor_No":[floor_no],
        "Total_Floors":[total_floors]
    })

    # OHE
    ohe_cols = [
        "Status",
        "Transaction",
        "Furnishing",
        "Facing",
        "City"
    ]

    ohe_data = ohe.transform(
        input_df[ohe_cols]
    )

    ohe_df = pd.DataFrame(
        ohe_data,
        columns=ohe.get_feature_names_out(ohe_cols)
    )

    # TARGET ENCODING

    te_df = te.transform(
        input_df[["Area_Name"]]
    )

    # NUMERICAL

    num_df = input_df[
        [
            "Bathroom",
            "Balcony",
            "BHK",
            "Area(sqft)",
            "Floor_No",
            "Total_Floors"
        ]
    ]

    scaled_num = std.transform(
        num_df
    )

    scaled_num_df = pd.DataFrame(
        scaled_num,
        columns=num_df.columns
    )

    # FINAL DATA

    final_df = pd.concat(
        [
            ohe_df.reset_index(drop=True),
            te_df.reset_index(drop=True),
            scaled_num_df.reset_index(drop=True)
        ],
        axis=1
    )

    # PREDICT

    prediction_log = model.predict(
        final_df
    )

    prediction_price = np.expm1(
        prediction_log
    )

    price = prediction_price[0]

    crores = price / 10000000
    lakhs = price / 100000
    
    # =====================================
    # CITY COMPARISON DATA
    # =====================================

    city_df = df[df["City"] == city]

    avg_price = city_df["Price_clean"].mean()

    difference = (
        (price - avg_price)
        / avg_price
    ) * 100

    # =====================================
    # DATASET STATS
    # =====================================

    mean_price = df["Price_clean"].mean()

    median_price = df["Price_clean"].median()

    max_price = df["Price_clean"].max()

    # =====================================
    # PRICE CARD
    # =====================================

    st.markdown("---")

    st.subheader("🏠 Predicted Property Price")

    price_container = st.container(border=True)

    with price_container:

        st.markdown(
            f"""
            # ₹ {crores:.2f} Crore
            """
        )

        st.write(
            f"₹ {lakhs:.2f} Lakhs"
        )

        st.write(
            f"₹ {price:,.0f}"
        )

    st.markdown("---")

    st.subheader("📈 Price Comparison")

    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric(
            "Average Price",
            f"₹ {avg_price/10000000:.2f} Cr"
        )

    with col2:
        st.metric(
            "Predicted Property",
            f"₹ {price/10000000:.2f} Cr"
        )

    with col3:
        st.metric(
            "Difference",
            f"{difference:.1f}%"
        )

    st.markdown("")

    if difference > 0:
        st.success(
            f"🟢 This property is {difference:.1f}% ABOVE the city average."
        )
    else:
        st.info(
            f"🔵 This property is {abs(difference):.1f}% BELOW the city average."
        )

    st.markdown("---")
    # -------------------------------------
    # Top Factors Affecting Price
    #-----------------------------------

    importance_df = pd.DataFrame({
    "Feature": final_df.columns,
    "Importance": model.feature_importances_
    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(10)
    )
    
    st.subheader("📊 Top Factors Affecting Price")

    st.bar_chart(
        importance_df.set_index("Feature")
    )

    # =====================================
    # PROPERTY SUMMARY
    # =====================================

    st.markdown("---")

    st.subheader("📋 Property Summary")

    c1, c2 = st.columns(2)

    with c1:
        st.info(
            f"""
    City : {city}

    Area : {area_name}

    BHK : {int(bhk)}
    """
        )

    with c2:
        st.info(
            f"""
    Bathroom : {int(bathroom)}

    Balcony : {int(balcony)}

    Area : {int(area)} sqft
    """
        )


    # =====================================
    # PHASE 5 : PROPERTY CATEGORY
    # =====================================
    st.markdown("---")

    st.subheader("🏆 Property Category")

    if price < 10000000:
        st.success("Budget Property 💰")

    elif price < 30000000:
        st.info("Premium Property ⭐")

    else:
        st.warning("Luxury Property 👑")

    

    st.markdown("---")
    st.subheader("📍 Property Location")
    city_coords = {
        "Delhi":[28.6139,77.2090],
        "Mumbai":[19.0760,72.8777],
        "Chennai":[13.0827,80.2707],
        "Hyderabad":[17.3850,78.4867],
        "Pune":[18.5204,73.8567]
    }

    lat, lon = city_coords[city]

    map_df = pd.DataFrame({
        "lat":[lat],
        "lon":[lon]
    })

    st.map(map_df)
    
    # =====================================
    # PHASE 4 : MODEL DETAILS
    # =====================================


    st.markdown("---")

    st.markdown("---")
    st.markdown("## 📊 Market Snapshot")

    c1,c2,c3 = st.columns(3)

    with c1:
        st.metric(
            "Average",
            f"₹ {mean_price/10000000:.2f} Cr"
        )

    with c2:
        st.metric(
            "Median",
            f"₹ {median_price/10000000:.2f} Cr"
        )

    with c3:
        st.metric(
            "Maximum",
            f"₹ {max_price/100000000:.2f} Cr"
        )

    st.markdown("---")

    with st.expander("🤖 Model Details"):
        st.write("Model : Random Forest")
        st.write("Features : 27")
        st.write("Records : 2877")
        st.write("Encoding : OHE + Target Encoding")
        st.write("Scaler : StandardScaler")

    st.markdown("---")

    st.caption(
    """
    🏠 Smart Property Valuation System

    Model : Random Forest Regressor

    Encoding : One Hot + Target Encoding

    Cities : Delhi, Mumbai, Chennai, Hyderabad, Pune

    Built using Streamlit + Machine Learning
    """
    )