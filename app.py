import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="UrbanFlow AI Dashboard", page_icon="🚦", layout="wide")

# ==========================================
# 1. MODEL TRAINING CACHE (Runs only once)
# ==========================================
@st.cache_resource
def load_and_train_model():
    # Load Data
    data_path = r"dataset\Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
    df = pd.read_csv(data_path)
    
    cols_to_keep = [
        'id', 'event_type', 'latitude', 'longitude', 'event_cause',
        'requires_road_closure', 'start_datetime', 'closed_datetime', 'priority', 'veh_type', 'description', 'zone', 'corridor'
    ]
    cols_present = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_present]
    
    # Preprocessing
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce', utc=True)
    df['closed_datetime'] = pd.to_datetime(df['closed_datetime'], errors='coerce', utc=True)
    df = df.dropna(subset=['start_datetime', 'closed_datetime']).copy()
    
    df['duration_mins'] = (df['closed_datetime'] - df['start_datetime']).dt.total_seconds() / 60.0
    df = df[(df['duration_mins'] >= 0) & (df['duration_mins'] <= 10080)].copy()
    
    def map_priority(p):
        p = str(p).lower()
        if 'high' in p: return 3
        if 'medium' in p: return 2
        return 1
    
    df['priority_score'] = df['priority'].apply(map_priority)
    df['requires_road_closure'] = df['requires_road_closure'].fillna(False).astype(bool)
    df['closure_multiplier'] = np.where(df['requires_road_closure'], 1.5, 1.0)
    
    df['raw_impact'] = df['duration_mins'] * df['priority_score'] * df['closure_multiplier']
    df['log_impact'] = np.log1p(df['raw_impact'])
    min_impact, max_impact = df['log_impact'].min(), df['log_impact'].max()
    df['impact_score'] = 1 + 9 * ((df['log_impact'] - min_impact) / (max_impact - min_impact + 1e-9))
    
    df['hour'] = df['start_datetime'].dt.hour
    df['is_weekend'] = df['start_datetime'].dt.dayofweek.isin([5, 6]).astype(int)
    
    def is_peak_hour(h):
        return 1 if (8 <= h <= 11) or (17 <= h <= 20) else 0
    df['is_peak_hour'] = df['hour'].apply(is_peak_hour)
    
    df = df[(df['latitude'] != 0) & (df['longitude'] != 0)].copy()
    df = df.dropna(subset=['latitude', 'longitude']).copy()
    
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    df['hotspot_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']])
    
    df['veh_type'] = df['veh_type'].fillna('unknown')
    df['zone'] = df['zone'].fillna('unknown')
    df['corridor'] = df['corridor'].fillna('unknown')
    df['event_cause'] = df['event_cause'].fillna('unknown')
    
    # Store unique categorical values for the UI Dropdowns before Dummies
    unique_vals = {
        'event_cause': df['event_cause'].unique().tolist(),
        'veh_type': df['veh_type'].unique().tolist(),
        'zone': df['zone'].unique().tolist(),
        'corridor': df['corridor'].unique().tolist()
    }
    
    df_model = pd.get_dummies(df, columns=['event_cause', 'veh_type', 'zone', 'corridor'], drop_first=True)
    
    df_model['description'] = df_model['description'].fillna('')
    tfidf = TfidfVectorizer(max_features=50, stop_words='english')
    desc_tfidf = tfidf.fit_transform(df_model['description']).toarray()
    tfidf_cols = [f'tfidf_{i}' for i in range(desc_tfidf.shape[1])]
    df_tfidf = pd.DataFrame(desc_tfidf, columns=tfidf_cols, index=df_model.index)
    df_model = pd.concat([df_model, df_tfidf], axis=1)
    
    base_features = ['hour', 'is_weekend', 'is_peak_hour', 'priority_score', 'closure_multiplier', 'hotspot_cluster']
    cat_features = [c for c in df_model.columns if 'event_cause_' in c or 'veh_type_' in c or 'zone_' in c or 'corridor_' in c]
    features = base_features + cat_features + tfidf_cols
    
    X = df_model[features]
    y = df_model['impact_score']
    
    # Fast Train using pre-discovered optimal hyperparameters (No RandomizedSearch needed here!)
    xgb_best = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    rf_best = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    
    model = VotingRegressor([('xgb', xgb_best), ('rf', rf_best)])
    model.fit(X, y)
    
    return model, kmeans, tfidf, features, unique_vals

# ==========================================
# 2. APP LAYOUT AND UI
# ==========================================
st.title("🚦 UrbanFlow AI: Event-Driven Congestion Predictor")
st.markdown("Instantly forecast traffic impact and issue prescriptive deployment orders.")

with st.spinner("Initializing AI Core..."):
    model, kmeans, tfidf, features, unique_vals = load_and_train_model()

st.sidebar.header("📝 Incident Parameters")

# Sidebar Inputs
col_1, col_2 = st.sidebar.columns(2)
in_hour = col_1.slider("Hour of Day", 0, 23, 17)
in_weekend = col_2.checkbox("Is Weekend?", value=False)
in_priority = st.sidebar.selectbox("Priority Level", ["Low", "Medium", "High"], index=2)
in_closure = st.sidebar.checkbox("Requires Road Closure", value=True)

st.sidebar.markdown("---")
in_lat = st.sidebar.number_input("Latitude", value=12.9716, format="%.6f")
in_lon = st.sidebar.number_input("Longitude", value=77.5946, format="%.6f")

st.sidebar.markdown("---")
in_cause = st.sidebar.selectbox("Event Cause", unique_vals['event_cause'])
in_veh = st.sidebar.selectbox("Vehicle Type", unique_vals['veh_type'])
in_zone = st.sidebar.selectbox("Zone", unique_vals['zone'])
in_corridor = st.sidebar.selectbox("Corridor", unique_vals['corridor'])
in_desc = st.sidebar.text_input("Incident Description", "heavy water logging blocking road")

# ==========================================
# 3. PREDICTION ENGINE
# ==========================================
if st.sidebar.button("🚀 Predict Impact & Prescribe Resources", type="primary"):
    
    # 3A. Feature Processing
    is_peak = 1 if (8 <= in_hour <= 11) or (17 <= in_hour <= 20) else 0
    p_score = 3 if in_priority == "High" else (2 if in_priority == "Medium" else 1)
    c_mult = 1.5 if in_closure else 1.0
    cluster = kmeans.predict(pd.DataFrame({'latitude': [in_lat], 'longitude': [in_lon]}))[0]
    
    input_data = {
        'hour': [in_hour],
        'is_weekend': [1 if in_weekend else 0],
        'is_peak_hour': [is_peak],
        'priority_score': [p_score],
        'closure_multiplier': [c_mult],
        'hotspot_cluster': [cluster],
        'event_cause': [in_cause],
        'veh_type': [in_veh],
        'zone': [in_zone],
        'corridor': [in_corridor]
    }
    
    df_input = pd.DataFrame(input_data)
    df_input_dummy = pd.get_dummies(df_input, columns=['event_cause', 'veh_type', 'zone', 'corridor'], drop_first=True)
    
    # Ensure all columns from training exist (fill missing dummies with 0)
    for col in features:
        if col not in df_input_dummy.columns and not col.startswith('tfidf_'):
            df_input_dummy[col] = 0
            
    # Process NLP
    desc_tfidf = tfidf.transform([in_desc]).toarray()
    for i in range(desc_tfidf.shape[1]):
        df_input_dummy[f'tfidf_{i}'] = desc_tfidf[0][i]
        
    X_pred = df_input_dummy[features]
    
    # 3B. Prediction
    predicted_score = model.predict(X_pred)[0]
    
    # 3C. Heuristic Matrix (from notebook)
    manpower = 0
    barricades = 0
    diversion = ""
    
    if predicted_score >= 8.0:
        manpower = 8
        barricades = 20
        diversion = "Major Area-Wide Diversion Required"
    elif predicted_score >= 5.0:
        manpower = 4
        barricades = 10
        diversion = "Local Block-Level Diversion"
    elif predicted_score >= 3.0:
        manpower = 2
        barricades = 5
        diversion = "Minor Lane Merging"
    else:
        manpower = 1
        barricades = 2
        diversion = "No Diversion (Warning signs only)"
        
    # Output to UI
    st.markdown("### 📊 AI Impact Prediction")
    
    # Dynamic styling based on severity
    if predicted_score >= 8.0:
        score_color = "red"
    elif predicted_score >= 5.0:
        score_color = "orange"
    else:
        score_color = "green"
        
    st.markdown(f"<h1 style='text-align: center; color: {score_color}; font-size: 80px;'>{predicted_score:.1f} / 10.0</h1>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🛠️ Prescriptive Deployment Orders")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("👮 Police Officers", f"{manpower} Units")
    col2.metric("🚧 Barricades", f"{barricades} Units")
    col3.metric("🗺️ Diversion Plan", diversion)
    
    st.success("Deployment orders generated successfully based on predicted impact severity.")
else:
    st.info("👈 Enter incident details in the sidebar and click 'Predict Impact' to generate deployment orders.")
