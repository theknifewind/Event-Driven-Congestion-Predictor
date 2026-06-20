import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pydeck as pdk
import datetime
import warnings
import dataclasses

warnings.filterwarnings('ignore')

# ==========================================
# 1. SKLEARN 1.6+ MONKEYPATCH
# ==========================================
if hasattr(xgb.XGBRegressor, '__sklearn_tags__'):
    orig_tags_method = xgb.XGBRegressor.__sklearn_tags__
    def patched_sklearn_tags(self):
        return dataclasses.replace(orig_tags_method(self), estimator_type="regressor")
    xgb.XGBRegressor.__sklearn_tags__ = patched_sklearn_tags

# Page setup
st.set_page_config(page_title="UrbanFlow AI - Operations Center", page_icon="🚦", layout="wide")

# ==========================================
# 2. PREMIUM CSS THEMING (Glassmorphism & Neon)
# ==========================================
st.markdown("""
<style>
    /* CSS Variables for theme consistency */
    :root {
        --bg-card: rgba(30, 41, 59, 0.45);
        --border-card: rgba(255, 255, 255, 0.08);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
    }
    
    /* Global layout tweaks */
    .stApp {
        background-color: #0b0f19;
    }
    
    /* Styling for glassmorphic KPI cards */
    .metric-card {
        background: var(--bg-card);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid var(--border-card);
        border-radius: 12px;
        padding: 22px;
        margin: 8px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.35);
        box-shadow: 0 6px 25px rgba(56, 189, 248, 0.15);
    }
    .metric-header {
        font-size: 11px;
        font-weight: 600;
        color: var(--text-secondary);
        letter-spacing: 1.5px;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .metric-val {
        font-size: 34px;
        font-weight: 700;
        color: var(--text-primary);
        margin-top: 10px;
        letter-spacing: -0.5px;
    }
    .metric-sub {
        font-size: 11px;
        color: #64748b;
        margin-top: 6px;
    }
    
    /* Dynamically colored badges */
    .status-badge {
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: inline-block;
        margin-top: 8px;
    }
    .badge-low {
        background-color: rgba(34, 197, 94, 0.12);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .badge-moderate {
        background-color: rgba(249, 115, 22, 0.12);
        color: #fb923c;
        border: 1px solid rgba(249, 115, 22, 0.3);
    }
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.12);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Primary AI score banner card */
    .score-container {
        text-align: center;
        padding: 30px 20px;
        background: radial-gradient(circle at top, rgba(30, 41, 59, 0.65), rgba(15, 23, 42, 0.85));
        border: 1px solid var(--border-card);
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        margin-bottom: 20px;
    }
    .score-title {
        font-size: 14px;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }
    
    /* Deployment layout box */
    .deploy-container {
        background: rgba(15, 23, 42, 0.4);
        border: 1px dashed rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 24px;
        margin-top: 15px;
    }
    .deploy-header {
        font-size: 16px;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Cost & Coverage tags */
    .summary-badge {
        font-size: 13px;
        font-weight: 600;
        color: var(--text-primary);
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 8px 16px;
        border-radius: 8px;
        text-align: center;
    }
    
    /* Glowing divider line */
    .glow-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.3), transparent);
        margin: 25px 0;
    }
    
    /* Sidebar styling tweaks */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid var(--border-card);
    }
</style>
""", unsafe_allow_html=True)

# Haversine distance calculator in numpy
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6367 * c

# ==========================================
# 3. MODEL TRAINING CACHE (Runs only once)
# ==========================================
@st.cache_resource
def load_and_train_model():
    data_path = "dataset/Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
    df = pd.read_csv(data_path)
    
    cols_to_keep = [
        'id', 'event_type', 'latitude', 'longitude', 'event_cause',
        'requires_road_closure', 'start_datetime', 'closed_datetime', 'end_datetime', 'priority', 'veh_type', 'description', 'zone', 'corridor'
    ]
    cols_present = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_present]
    
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce', utc=True)
    df['closed_datetime'] = pd.to_datetime(df['closed_datetime'], errors='coerce', utc=True)
    df['end_datetime'] = pd.to_datetime(df['end_datetime'], errors='coerce', utc=True)
    
    # Coalesce resolution times (end_datetime for planned, closed_datetime for unplanned)
    df['resolved_time'] = df['closed_datetime'].fillna(df['end_datetime'])
    df = df.dropna(subset=['start_datetime', 'resolved_time']).copy()
    
    df['duration_mins'] = (df['resolved_time'] - df['start_datetime']).dt.total_seconds() / 60.0
    df = df[(df['duration_mins'] >= 0) & (df['duration_mins'] <= 10080)].copy()
    
    # Clean spatial coordinates
    df = df[(df['latitude'] != 0) & (df['longitude'] != 0)].copy()
    df = df.dropna(subset=['latitude', 'longitude']).copy()
    
    # Calculate Spatial-Temporal Concurrent Overlapping Load (Real-time signal)
    lats = df['latitude'].values
    lons = df['longitude'].values
    starts = df['start_datetime'].values
    resolveds = df['resolved_time'].values
    n = len(df)
    active_loads = [np.sum(haversine_np(lons[i], lats[i], lons[(starts <= starts[i]) & (resolveds >= starts[i])], lats[(starts <= starts[i]) & (resolveds >= starts[i])]) <= 5.0) - 1 for i in range(n)]
    df['nearby_active_events'] = active_loads
    
    # Encode event_type (planned vs unplanned)
    df['event_type_planned'] = (df['event_type'].str.lower() == 'planned').astype(int)
    
    def map_priority(p):
        p = str(p).lower()
        if 'high' in p: return 3
        if 'medium' in p: return 2
        return 1
    
    df['priority_score'] = df['priority'].apply(map_priority)
    df['requires_road_closure'] = df['requires_road_closure'].fillna(False).astype(bool)
    df['closure_multiplier'] = np.where(df['requires_road_closure'], 1.5, 1.0)
    
    # Calculate synthetic target bounds for scaling downstream
    df['raw_impact'] = df['duration_mins'] * df['priority_score'] * df['closure_multiplier']
    df['log_impact'] = np.log1p(df['raw_impact'])
    min_impact, max_impact = df['log_impact'].min(), df['log_impact'].max()
    
    df['hour'] = df['start_datetime'].dt.hour
    df['is_weekend'] = df['start_datetime'].dt.dayofweek.isin([5, 6]).astype(int)
    
    def is_peak_hour(h):
        return 1 if (8 <= h <= 11) or (17 <= h <= 20) else 0
    df['is_peak_hour'] = df['hour'].apply(is_peak_hour)
    
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    df['hotspot_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']])
    
    df['veh_type'] = df['veh_type'].fillna('unknown')
    df['zone'] = df['zone'].fillna('unknown')
    df['corridor'] = df['corridor'].fillna('unknown')
    df['event_cause'] = df['event_cause'].fillna('unknown')
    
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
    
    base_features = [
        'hour', 'is_weekend', 'is_peak_hour', 'hotspot_cluster',
        'nearby_active_events', 'priority_score', 'closure_multiplier', 'event_type_planned'
    ]
    cat_features = [c for c in df_model.columns if 'event_cause_' in c or 'veh_type_' in c or 'zone_' in c or 'corridor_' in c]
    features = base_features + cat_features + tfidf_cols
    
    X = df_model[features]
    y = np.log1p(df_model['duration_mins']) # Leakage-free Target variable: log of duration
    
    xgb_best = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    rf_best = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    
    model = VotingRegressor([('xgb', xgb_best), ('rf', rf_best)])
    model.fit(X, y)
    
    return model, kmeans, tfidf, features, unique_vals, min_impact, max_impact

# Initialize model
with st.spinner("Initializing AI Engine..."):
    model, kmeans, tfidf, features, unique_vals, min_impact, max_impact = load_and_train_model()

# ==========================================
# 4. INITIALIZE SESSION STATE
# ==========================================
if 'logged_events' not in st.session_state:
    st.session_state.logged_events = []
if 'active_prediction' not in st.session_state:
    st.session_state.active_prediction = None

# ==========================================
# 5. PREDICTION & EXTRA METRICS CALCULATOR
# ==========================================
def calculate_metrics(in_event_type, in_hour, in_weekend, in_priority, in_closure, in_lat, in_lon, in_cause, in_veh, in_zone, in_corridor, in_desc):
    is_peak = 1 if (8 <= in_hour <= 11) or (17 <= in_hour <= 20) else 0
    p_score = 3 if in_priority == "High" else 1
    c_mult = 1.5 if in_closure else 1.0
    cluster = kmeans.predict(pd.DataFrame({'latitude': [in_lat], 'longitude': [in_lon]}))[0]
    
    # Calculate live active overlapping events (Real-time signal)
    nearby_count = 0
    for event in st.session_state.logged_events:
        if event['status'] != 'RESOLVED' and 'inputs' in event:
            d = haversine_np(in_lon, in_lat, event['inputs']['lon'], event['inputs']['lat'])
            if d <= 5.0:
                nearby_count += 1
                
    input_data = {
        'hour': [in_hour],
        'is_weekend': [1 if in_weekend else 0],
        'is_peak_hour': [is_peak],
        'hotspot_cluster': [cluster],
        'nearby_active_events': [nearby_count],
        'priority_score': [p_score],
        'closure_multiplier': [c_mult],
        'event_type_planned': [1 if in_event_type == "planned" else 0],
        'event_cause': [in_cause],
        'veh_type': [in_veh],
        'zone': [in_zone],
        'corridor': [in_corridor]
    }
    
    df_input = pd.DataFrame(input_data)
    df_input_dummy = pd.get_dummies(df_input, columns=['event_cause', 'veh_type', 'zone', 'corridor'], drop_first=True)
    
    for col in features:
        if col not in df_input_dummy.columns and not col.startswith('tfidf_'):
            df_input_dummy[col] = 0
            
    desc_tfidf = tfidf.transform([in_desc]).toarray()
    for i in range(desc_tfidf.shape[1]):
        df_input_dummy[f'tfidf_{i}'] = desc_tfidf[0][i]
        
    X_pred = df_input_dummy[features]
    
    # Predict duration log
    pred_log_duration = float(model.predict(X_pred)[0])
    pred_duration = np.expm1(pred_log_duration)
    
    # Downstream policy business rule to calculate impact score
    raw_impact = pred_duration * p_score * c_mult
    log_raw_impact = np.log1p(raw_impact)
    
    # Scale score to 1.0 - 10.0 range
    predicted_score = float(1.0 + 9.0 * ((log_raw_impact - min_impact) / (max_impact - min_impact + 1e-9)))
    predicted_score = max(1.0, min(10.0, predicted_score))
    
    # 1. Affected Radius (km) - Prescriptive Heuristic
    priority_coef = 0.5 if in_priority == "Low" else 2.2
    closure_coef = 1.6 if in_closure else 1.0
    desc_weight = min(len(in_desc) * 0.005, 0.3)
    affected_radius = round((0.8 + priority_coef + desc_weight) * closure_coef * 0.8, 2)
    
    # 2. Road Saturation (%) - Prescriptive Heuristic
    base_sat = 12.0
    peak_sat = 22.0 if is_peak else 5.0
    priority_sat = 5.0 if in_priority == "Low" else 38.0
    closure_sat = 20.0 if in_closure else 2.0
    road_saturation = round(min(99.0, base_sat + peak_sat + priority_sat + closure_sat + (predicted_score * 1.5)), 1)
    
    # 3. Est. Delay (mins) - Prescriptive Heuristic
    est_delay = int((predicted_score ** 1.85) * (1.35 if is_peak else 0.75) + (25 if in_closure else 0))
    if predicted_score < 4.0 and not in_closure:
        est_delay = max(0, int(predicted_score - 3))
        
    # 4. Traffic Increase (%) - Prescriptive Heuristic
    traffic_increase = round(min(160.0, (predicted_score * 13) + (35 if is_peak else 8) + (25 if in_closure else 4)), 1)
    
    # 5. Optimized Deployment (Matching Teammate-level high-fidelity scaling)
    zone_factor = 1.25 if "Zone 2" in in_zone or "Zone 1" in in_zone else 1.0
    corridor_factor = 1.3 if in_corridor != "Non-corridor" else 0.85
    base_units = 18 if in_priority == "Low" else 85
    peak_factor = 1.35 if is_peak else 0.8
    closure_factor = 1.45 if in_closure else 1.0
    
    officers = int(base_units * peak_factor * closure_factor * zone_factor * corridor_factor * (0.45 + predicted_score / 14.0))
    officers = max(5, min(150, officers))
    patrol_vehicles = int(max(1, officers * 0.24 + 0.5))
    barricades = int(officers * 0.38 + (12 if in_closure else 2))
    barricades = max(2, min(80, barricades))
    
    # Cost (Matching ₹2,500/Officer, ₹1,200/Vehicle, ₹270/Barricade)
    total_cost = (officers * 2500) + (patrol_vehicles * 1200) + (barricades * 270)
    
    # Coverage Calculation
    required_officers = int(officers * (1.0 + (10.1 - predicted_score) * 0.065))
    coverage = round((officers / max(1, required_officers)) * 100, 1)
    coverage = min(99.5, coverage)
    
    if predicted_score >= 7.5:
        diversion = "Major Area-Wide Diversion Required"
    elif predicted_score >= 4.5:
        diversion = "Local Block-Level Diversion"
    elif predicted_score >= 3.0:
        diversion = "Minor Lane Merging"
    else:
        diversion = "No Diversion (Warning signs only)"
        
    return {
        'score': predicted_score,
        'radius': affected_radius,
        'saturation': road_saturation,
        'delay': est_delay,
        'increase': traffic_increase,
        'officers': officers,
        'vehicles': patrol_vehicles,
        'barricades': barricades,
        'cost': total_cost,
        'coverage': coverage,
        'diversion': diversion,
        'is_peak': is_peak,
        'p_score': p_score,
        'inputs': {
            'event_type': in_event_type,
            'hour': in_hour,
            'weekend': in_weekend,
            'priority': in_priority,
            'closure': in_closure,
            'lat': in_lat,
            'lon': in_lon,
            'cause': in_cause,
            'veh': in_veh,
            'zone': in_zone,
            'corridor': in_corridor,
            'desc': in_desc
        }
    }

# ==========================================
# 6. DIGITAL TWIN SIMULATION COMPONENT (HTML/CSS)
# ==========================================
def get_digital_twin_html(score, saturation):
    if score >= 7.5:
        flow_speed = "28s"
        car_color = "#f87171"
        density_label = "GRIDLOCK"
        car_count = 15
    elif score >= 4.5:
        flow_speed = "14s"
        car_color = "#fb923c"
        density_label = "HEAVY"
        car_count = 10
    else:
        flow_speed = "4.5s"
        car_color = "#4ade80"
        density_label = "FREE FLOW"
        car_count = 5

    cars_html = ""
    for i in range(car_count):
        delay = i * (1.6 if score >= 7.5 else (0.9 if score >= 4.5 else 0.5))
        lane = (i % 3) + 1
        cars_html += f'<div class="car lane-{lane}" style="animation-duration: {flow_speed}; animation-delay: {delay}s; background: {car_color};">🚗</div>'

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
      body {{
        background: #0f172a;
        margin: 0;
        padding: 10px;
        font-family: 'Inter', sans-serif;
        overflow: hidden;
      }}
      .road-container {{
        position: relative;
        width: 100%;
        height: 220px;
        background: #1e293b;
        border-radius: 12px;
        border: 2px solid #334155;
        box-sizing: border-box;
      }}
      .road-lines {{
        position: absolute;
        width: 100%;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-evenly;
        pointer-events: none;
      }}
      .line {{
        width: 100%;
        height: 4px;
        background: #fef08a;
        border-top: 2px dashed #94a3b8;
      }}
      .sensor-overlay {{
        position: absolute;
        top: 0;
        left: 25%;
        width: 50%;
        height: 100%;
        background: rgba(56, 189, 248, 0.05);
        border-left: 2px dashed rgba(56, 189, 248, 0.25);
        border-right: 2px dashed rgba(56, 189, 248, 0.25);
        display: flex;
        align-items: center;
        justify-content: center;
        color: rgba(56, 189, 248, 0.5);
        font-size: 11px;
        letter-spacing: 2px;
        text-transform: uppercase;
      }}
      .car {{
        position: absolute;
        width: 34px;
        height: 20px;
        border-radius: 5px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.5);
        animation: drive linear infinite;
      }}
      .lane-1 {{ top: 32px; }}
      .lane-2 {{ top: 100px; }}
      .lane-3 {{ top: 168px; }}
      
      @keyframes drive {{
        0% {{ left: -60px; }}
        100% {{ left: 100%; }}
      }}
      .info-header {{
        display: flex;
        justify-content: space-between;
        color: #94a3b8;
        font-size: 12px;
        margin-bottom: 8px;
        padding: 0 4px;
      }}
      .status-tag {{
        background: {car_color}22;
        color: {car_color};
        padding: 2px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 10px;
        border: 1px solid {car_color}44;
      }}
    </style>
    </head>
    <body>
      <div class="info-header">
        <span>🚦 Digital Twin Corridor Simulation</span>
        <span class="status-tag">STATUS: {density_label} ({saturation}%)</span>
      </div>
      <div class="road-container">
        <div class="road-lines">
          <div class="line"></div>
          <div class="line"></div>
        </div>
        <div class="sensor-overlay">DIGITAL TWIN SENSORS</div>
        {cars_html}
      </div>
    </body>
    </html>
    """
    return html_code

# ==========================================
# 8. APP LAYOUT
# ==========================================
st.title("🚦 UrbanFlow AI: Command & Operations Center")
st.markdown("Instantly forecast traffic impact, simulate digital twin corridors, and issue prescriptive deployment orders.")

# Sidebar Parameters
st.sidebar.header("📝 Incident Parameters")

in_event_type = st.sidebar.selectbox("Event Type", ["unplanned", "planned"], index=0)

col_1, col_2 = st.sidebar.columns(2)
in_hour = col_1.slider("Hour of Day", 0, 23, 17)
in_weekend = col_2.checkbox("Is Weekend?", value=False)
in_priority = st.sidebar.selectbox("Priority Level", ["Low", "High"], index=1)
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

# Automatically compute prediction and save to session state when sidebar inputs change
pred_res = calculate_metrics(in_event_type, in_hour, in_weekend, in_priority, in_closure, in_lat, in_lon, in_cause, in_veh, in_zone, in_corridor, in_desc)
st.session_state.active_prediction = pred_res

# Main Dashboard Navigation Tabs
tab_cc, tab_twin, tab_manager = st.tabs([
    "🚨 Command Center", 
    "🔀 Digital Twin & Live Map", 
    "📋 Event Manager"
])

# ==========================================
# TAB 1: COMMAND CENTER
# ==========================================
with tab_cc:
    res = st.session_state.active_prediction
    
    col_left, col_right = st.columns([1, 1.2])
    
    with col_left:
        # Score Container
        score = res['score']
        if score >= 7.5:
            badge_class = "badge-critical"
            badge_label = "CRITICAL"
        elif score >= 4.5:
            badge_class = "badge-moderate"
            badge_label = "MODERATE"
        else:
            badge_class = "badge-low"
            badge_label = "LOW"
            
        st.markdown(f"""
        <div class="score-container">
            <div class="score-title">Congestion Impact Score</div>
            <div style="font-size: 72px; font-weight: 800; color: #38bdf8; margin: 15px 0;">{score:.2f} / 10</div>
            <div class="status-badge {badge_class}">{badge_label}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 2x2 Grid of Metrics
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-header">📍 Affected Radius</div>
                <div class="metric-val">{res['radius']} km</div>
                <div class="metric-sub">Estimated spillover zone</div>
            </div>
            <div class="metric-card">
                <div class="metric-header">⏱️ Est. Delay</div>
                <div class="metric-val">+{res['delay']}m</div>
                <div class="metric-sub">Average travel time loss</div>
            </div>
            """, unsafe_allow_html=True)
            
        with metric_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-header">🚗 Road Saturation</div>
                <div class="metric-val">{res['saturation']}%</div>
                <div class="metric-sub">Active carrying capacity</div>
            </div>
            <div class="metric-card">
                <div class="metric-header">📈 Traffic Increase</div>
                <div class="metric-val">{res['increase']}%</div>
                <div class="metric-sub">Diversion path volume spike</div>
            </div>
            """, unsafe_allow_html=True)
            
    with col_right:
        # Optimized Deployment Card
        st.markdown(f"""
        <div class="deploy-container">
            <div class="deploy-header">👮 Optimized Resource Deployment</div>
            <div style="display: flex; flex-direction: column; gap: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.03); padding: 12px 18px; border-radius: 8px; border: 1px solid var(--border-card);">
                    <span style="font-size: 14px; color: var(--text-secondary);">Police Officers</span>
                    <span style="font-size: 20px; font-weight: 700; color: #38bdf8;">{res['officers']} Units</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.03); padding: 12px 18px; border-radius: 8px; border: 1px solid var(--border-card);">
                    <span style="font-size: 14px; color: var(--text-secondary);">Patrol Vehicles</span>
                    <span style="font-size: 20px; font-weight: 700; color: #a855f7;">{res['vehicles']} Units</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.03); padding: 12px 18px; border-radius: 8px; border: 1px solid var(--border-card);">
                    <span style="font-size: 14px; color: var(--text-secondary);">Barricades</span>
                    <span style="font-size: 20px; font-weight: 700; color: #eab308;">{res['barricades']} Units</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.03); padding: 12px 18px; border-radius: 8px; border: 1px solid var(--border-card);">
                    <span style="font-size: 14px; color: var(--text-secondary);">Diversion Strategy</span>
                    <span style="font-size: 14px; font-weight: 700; color: #f8fafc;">{res['diversion']}</span>
                </div>
            </div>
            <div style="display: flex; gap: 15px; margin-top: 25px;">
                <div class="summary-badge" style="flex: 1;">💵 Cost: ₹{res['cost']:,}</div>
                <div class="summary-badge" style="flex: 1;">🛡️ Coverage: {res['coverage']}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Log to event manager button
        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
        if st.button("📥 Dispatch Units & Log to Event Manager", type="primary", use_container_width=True):
            event_id = f"EV-{int(datetime.datetime.now().timestamp()) % 100000}"
            event_log = {
                'id': event_id,
                'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'desc': in_desc,
                'priority': in_priority,
                'score': res['score'],
                'officers': res['officers'],
                'vehicles': res['vehicles'],
                'barricades': res['barricades'],
                'cost': res['cost'],
                'status': 'DISPATCHED',
                'inputs': {
                    'event_type': in_event_type,
                    'hour': in_hour,
                    'weekend': in_weekend,
                    'priority': in_priority,
                    'closure': in_closure,
                    'lat': in_lat,
                    'lon': in_lon,
                    'cause': in_cause,
                    'veh': in_veh,
                    'zone': in_zone,
                    'corridor': in_corridor,
                    'desc': in_desc
                }
            }
            st.session_state.logged_events.append(event_log)
            st.success(f"Incident logged successfully with ID: {event_id} under DISPATCHED status.")
            
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        st.caption("⚠️ Operational Heuristics Disclaimer: Metrics like Road Saturation, Affected Radius, Est. Delay, and Traffic Increase are computed via prescriptive heuristics derived from priority, closure status, and predicted impact. They are not direct predictions from the machine learning model.")


# ==========================================
# TAB 2: DIGITAL TWIN & LIVE MAP
# ==========================================
with tab_twin:
    res = st.session_state.active_prediction
    
    col_map, col_twin = st.columns([1, 1])
    
    with col_map:
        st.markdown("### 🗺️ Live Congestion Map Overlay")
        # Pydeck Map
        map_df = pd.DataFrame({
            'latitude': [res['inputs']['lat']],
            'longitude': [res['inputs']['lon']],
            'radius': [res['radius'] * 1000] # meters
        })
        
        view_state = pdk.ViewState(
            latitude=res['inputs']['lat'],
            longitude=res['inputs']['lon'],
            zoom=12.5,
            pitch=40
        )
        
        color_fill = [239, 68, 68, 80] if res['score'] >= 7.5 else ([251, 146, 60, 80] if res['score'] >= 4.5 else [74, 222, 128, 80])
        
        circle_layer = pdk.Layer(
            "ScatterplotLayer",
            map_df,
            get_position=["longitude", "latitude"],
            get_radius="radius",
            get_fill_color=color_fill,
            pickable=True
        )
        
        pin_layer = pdk.Layer(
            "ScatterplotLayer",
            map_df,
            get_position=["longitude", "latitude"],
            get_radius=120,
            get_fill_color=[255, 255, 255, 240],
            get_line_color=[15, 23, 42, 255],
            line_width_min_pixels=2,
            pickable=True
        )
        
        r = pdk.Deck(
            layers=[circle_layer, pin_layer],
            initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/dark-v9"
        )
        st.pydeck_chart(r)
        
    with col_twin:
        st.markdown("### 🔀 Active Digital Twin Simulation")
        # Digital Twin road simulation using HTML iframe
        st.components.v1.html(
            get_digital_twin_html(res['score'], res['saturation']),
            height=270,
            scrolling=False
        )
        
        st.markdown("""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid var(--border-card); border-radius: 8px; padding: 15px; font-size: 13px; color: var(--text-secondary);">
            <strong>Sensor Telemetry Status:</strong> Active. 
            Digital Twin represents real-time vehicle flow on regional lanes based on priority constraints and predicted saturation.
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 4: EVENT MANAGER
# ==========================================
with tab_manager:
    st.markdown("### 📋 Active Incident Event Log")
    
    if not st.session_state.logged_events:
        st.info("No active incidents currently logged. Go to the 'Command Center' tab and click 'Dispatch Units & Log' to save simulated events.")
    else:
        # Render logged events as cards
        for idx, event in enumerate(st.session_state.logged_events):
            event_id = event['id']
            
            # Colored indicator bar based on score
            score = event['score']
            if score >= 7.5:
                border_color = "#ef4444"
                status_color = "red"
            elif score >= 4.5:
                border_color = "#fb923c"
                status_color = "orange"
            else:
                border_color = "#22c55e"
                status_color = "green"
                
            col_a, col_b, col_c = st.columns([2.5, 1.5, 1])
            
            with col_a:
                st.markdown(f"""
                <div style="border-left: 5px solid {border_color}; background: rgba(30,41,59,0.3); border-radius: 4px; padding: 12px; margin: 10px 0;">
                    <div style="font-weight: 700; color: #38bdf8;">Incident {event_id} ({event['time']})</div>
                    <div style="font-size: 13px; margin-top: 5px; color: var(--text-primary);"><strong>Desc:</strong> {event['desc']}</div>
                    <div style="font-size: 12px; margin-top: 3px; color: var(--text-secondary);">
                        Priority: {event['priority']} | Score: {event['score']:.2f}/10
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_b:
                st.markdown(f"""
                <div style="background: rgba(30,41,59,0.3); border-radius: 4px; padding: 12px; margin: 10px 0; font-size: 12px; color: var(--text-secondary);">
                    <strong>Resources:</strong><br>
                    👮 {event['officers']} Officers | 🚓 {event['vehicles']} Vehicles | 🚧 {event['barricades']} Barricades<br>
                    <strong>Cost:</strong> ₹{event['cost']:,}
                </div>
                """, unsafe_allow_html=True)
                
            with col_c:
                # Dispatch control dropdown and actions
                new_status = st.selectbox(
                    "Status", 
                    ["DISPATCHED", "IN PROGRESS", "RESOLVED"], 
                    index=["DISPATCHED", "IN PROGRESS", "RESOLVED"].index(event['status']),
                    key=f"status_{event_id}"
                )
                
                # Update status if changed
                if new_status != event['status']:
                    st.session_state.logged_events[idx]['status'] = new_status
                    st.success(f"Status of {event_id} updated to {new_status}!")
                    st.rerun()
                    
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            
        # Button to clear resolved events
        if st.button("🧹 Clear Resolved Incidents", use_container_width=True):
            st.session_state.logged_events = [e for e in st.session_state.logged_events if e['status'] != 'RESOLVED']
            st.success("Cleared all resolved events!")
            st.rerun()
