import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
import plotly.express as px
import os, io, base64, requests

st.set_page_config(layout='wide')

#########################################
### Functions
#########################################

@st.cache_data
def load_data_for_instrument(instrument: str) -> pd.DataFrame:
    owner  = "TuckerArrants"
    repo   = "sessions_private"
    branch = "main"
    path   = f"{instrument}_Session_Data_Final_From_2008_V2.csv" 

    api_url = (
        f"https://api.github.com/repos/{owner}/{repo}"
        f"/contents/{path}?ref={branch}"
    )
    token   = st.secrets["GITHUB_TOKEN"]
    headers = {"Authorization": f"Bearer {token}"}

    resp = requests.get(api_url, headers=headers)
    if resp.status_code != 200:
        st.error(f"Couldn’t load {path}: HTTP {resp.status_code}")
        return pd.DataFrame()

    meta = resp.json()
    b64  = meta.get("content") or ""
    if not b64:
        sha      = meta.get("sha")
        blob_url = f"https://api.github.com/repos/{owner}/{repo}/git/blobs/{sha}"
        blob_resp = requests.get(blob_url, headers=headers)
        if blob_resp.status_code != 200:
            st.error(f"Couldn’t load blob {sha}: HTTP {blob_resp.status_code}")
            return pd.DataFrame()
        b64 = blob_resp.json().get("content", "")

    if not b64.strip():
        st.error(f"No content found in `{path}`")
        return pd.DataFrame()

    raw = base64.b64decode(b64).decode("utf-8")
    try:
        return pd.read_csv(io.StringIO(raw))
    except pd.errors.EmptyDataError:
        st.error(f"{path} has no parseable rows.")
        return pd.DataFrame()

def add_open_vs_flags(df):
    df = df.copy()
    base = df["open_1800"]
    for seg in ("adr","odr","rdr"):
        seg_open = df[f"{seg}_open"]
        # ← use the exact suffix your filters expect
        df[f"{seg}_open_to_1800_open"] = np.select(
            [seg_open > base, seg_open < base],
            ["Above","Below"],
            default="Neither",
        )
    return df

def add_rdr_open_vs_gap_flags(df):
    df = df.copy()
    for seg in ("adr", "odr", "rdr"):
        seg_open = df[f"{seg}_open"]
        base     = df["prev_close_1555"]
        df[f"{seg}_open_to_prev_1555_close"] = np.select(
            [seg_open > base, seg_open < base],
            ["Above", "Below"],
            default="Neither",
        )
    return df

def extract_time(df, time_cols=None):
    df = df.copy()
    if time_cols is None:
        time_cols = ["prev_rdr_conf_time", "adr_conf_time", "odr_conf_time", "rdr_conf_time",
                     'pre_adr_high_time', 'pre_adr_low_time', 'adr_high_time','adr_low_time',
                     'adr_transition_high_time', 'adr_transition_low_time',
                     'odr_transition_high_time', 'odr_transition_low_time',
                     'odr_high_time', 'odr_low_time', 'rdr_high_time', 'rdr_low_time',]

    for col in time_cols:
        # 1) coerce to datetime (NaT on failure)
        df[col] = pd.to_datetime(df[col], errors="coerce")
        # 2) extract HH:MM, empty string if missing
        df[f"{col}_hm"] = df[col].dt.strftime("%H:%M").fillna("")
    return df

def bucket_prev_close_touch(df,
                            touch_col="prev_close_1555_rdr_touch",
                            conf_col="rdr_conf_time"):
    df = df.copy()
    # 1) parse to full datetimes
    df[touch_col] = pd.to_datetime(df[touch_col], errors="coerce")
    df[conf_col]  = pd.to_datetime(df[conf_col],  errors="coerce")

    # 2) infer session key from the conf_col name
    #    (assumes 'rdr_conf_time', 'odr_conf_time', or 'adr_conf_time')
    session = None
    for s in ("rdr","odr","adr"):
        if conf_col.startswith(f"{s}_"):
            session = s
            break
    if session is None:
        raise ValueError(f"Cannot infer session from '{conf_col}'")

    # 3) map session → threshold hour (in fractional hours)
    threshold_map = {
        "rdr": 10 + 30/60,   # 10.5
        "odr": 4  +   0/60,  #  4.0
        "adr": 20 + 30/60,   # 20.5 (8:30pm)
    }
    threshold = threshold_map[session]

    # 4) convert each timestamp into a floating‐point “hour of day”
    def to_float_hour(ts):
        return ts.dt.hour + ts.dt.minute/60 + ts.dt.second/3600

    touch_hr = to_float_hour(df[touch_col])
    conf_hr  = to_float_hour(df[conf_col])

    # 5) if this is ADR (19:30 → 02:00), bump any post‐midnight times up by +24h
    #    so that 00:30 becomes 24.5 rather than 0.5
    if session == "adr":
        session_start = 19 + 30/60   # 19.5 == 19:30
        touch_hr = np.where(touch_hr < session_start, touch_hr + 24, touch_hr)
        conf_hr  = np.where(conf_hr  < session_start,  conf_hr  + 24, conf_hr)

    # 6) build your three buckets
    conds = [
        touch_hr < threshold,
        (touch_hr >= threshold) & (touch_hr < conf_hr),
        touch_hr >= conf_hr,
    ]
    choices = ["box_formation", "before_confirmation", "after_confirmation"]

    df[f"{touch_col}_time_buckets_v2"] = np.select(
        conds,
        choices,
        default=None,
    )

    return df

def drop_cols(df):
    to_drop = ['pre_adr_low_touch', 'prev_rdr_low_touch', 'adr_low_touch', 'adr_transition_low_touch', 'odr_low_touch', 'odr_transition_low_touch',
              'pre_adr_high_touch', 'prev_rdr_high_touch', 'adr_high_touch', 'adr_transition_high_touch', 'odr_high_touch', 'odr_transition_high_touch',
               
              'prev_rdr_idr_midline_touch', 'adr_idr_midline_touch', 'odr_idr_midline_touch',
               
              'prev_rdr_open_touch', 'adr_open_touch', 'odr_open_touch',
              'prev_rdr_close_touch', 'adr_close_touch', 'odr_close_touch',

               'pre_adr_high', 'pre_adr_low', 'adr_high', 'adr_low', 'adr_transition_high', 'adr_transition_low',
               'odr_high', 'odr_low', 'odr_transition_low', 'odr_transition_high', 'rdr_high', 'rdr_low',
               'prev_rdr_high', 'prev_rdr_low',
              
              'prev_close_1555_adr_touch', 'prev_close_1555_odr_touch', 'prev_close_1555_rdr_touch',
              'open_1800_adr_touch', 'open_1800_odr_touch', 'open_1800_rdr_touch']

    existing = [col for col in to_drop if col in df.columns]
    return df.drop(columns=existing)

def get_hod_lod(df):
    high_cols = [c for c in df.columns if c.endswith('_high') and "prev" not in c]
    low_cols  = [c for c in df.columns if c.endswith('_low')  and "prev" not in c]

    # 1) 5m HOD/LOD price
    df['hod_price'] = df[high_cols].max(axis=1)
    df['lod_price'] = df[low_cols].min(axis=1)

    # 2) which segment gave us that HOD/LOD?
    df['hod'] = (
        df[high_cols]
        .idxmax(axis=1)
        .str.replace(r'_high$', '', regex=True)
    )
    df['lod'] = (
        df[low_cols]
        .idxmin(axis=1)
        .str.replace(r'_low$',  '', regex=True)
    )

    df['hod_hm'] = df.apply(
        lambda row: row[f"{row['hod']}_high_time_hm"],
        axis=1
    )
    df['lod_hm'] = df.apply(
        lambda row: row[f"{row['lod']}_low_time_hm"],
        axis=1
    )

    return df

# ✅ Store username-password pairs
USER_CREDENTIALS = {
    "badboyz": "bangbang",
    "dreamteam" : "strike",
}

segments = {
    "Daily Open-ADR Transition":        (   0,  90),
    "ADR":            (  90, 480),
    "ADR-ODR Transition": ( 480, 540),
    "ODR":            ( 540, 870),
    "ODR-RDR Transition": ( 870, 930),
    "RDR":            ( 930,1380),
}

#########################################
### Log In
#########################################
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None

if not st.session_state["authenticated"]:
    st.title("Login to Database")

    # Username and password fields
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")

    # Submit button
    if st.button("Login"):
        if username in USER_CREDENTIALS and password == USER_CREDENTIALS[username]:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username  # Store the username
            # ← Clear *all* @st.cache_data caches here:
            st.cache_data.clear()

            st.success(f"Welcome, {username}! Loading fresh data…")
            st.rerun()
        else:
            st.error("Incorrect username or password. Please try again.")

    # Stop execution if user is not authenticated
    st.stop()

# ✅ If authenticated, show the full app
st.title("Trompete Kostet Knete")

# ↓ in your sidebar:
instrument_options = ["ES", "NQ", "YM", "CL", "GC", "NG", "HG", "SI", "ZN"]
selected_instrument = st.sidebar.selectbox("Instrument", instrument_options)

#########################################
### Data Loading and Processing
#########################################
df = load_data_for_instrument(selected_instrument)

df['date'] = pd.to_datetime(df['session_date']).dt.date
df = extract_time(df)
df = get_hod_lod(df)
df = add_open_vs_flags(df)
df = add_rdr_open_vs_gap_flags(df)

df = bucket_prev_close_touch(df, touch_col='prev_close_1555_adr_touch', conf_col='adr_conf_time')
df = bucket_prev_close_touch(df, touch_col='prev_close_1555_odr_touch', conf_col='odr_conf_time')
df = bucket_prev_close_touch(df, touch_col='prev_close_1555_rdr_touch', conf_col='rdr_conf_time')

df = bucket_prev_close_touch(df, touch_col='open_1800_adr_touch', conf_col='adr_conf_time')
df = bucket_prev_close_touch(df, touch_col='open_1800_odr_touch', conf_col='odr_conf_time')
df = bucket_prev_close_touch(df, touch_col='open_1800_rdr_touch', conf_col='rdr_conf_time')

df = drop_cols(df)

rename_map = {'pre_adr' : 'Daily Open-ADR Transition',
              'adr' : 'ADR',
              'adr_transition' : 'ADR-ODR Transition',
              'odr' : 'ODR',
              'odr_transition' : 'ODR-RDR Transition',
              'rdr' : 'RDR',
              'untouched' : 'Untouched',
              'uxp' : 'UXP',
              'ux' : 'UX',
              'u' : 'U',
              'dxp' : 'DXP',
              'dx' : 'DX',
              'd' : 'D',
              'rx' : 'RX',
              'rc' : 'RC',
              'none' : 'None',
              'long' : 'Long',
              'short' : 'Short',   
              'box_formation' : 'Box Formation',
              'before_confirmation' : 'Before Confirmation',
              'after_confirmation' : 'After Confirmation',
} 

df = df.replace(rename_map)

# 1) Make sure 'date' is a datetime column
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["session_date"])
else:
    st.sidebar.warning("No 'date' column found in your data!")

#########################################
### Sidebar
#########################################
day_options = ['All'] + ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
selected_day = st.sidebar.selectbox("Day of Week", day_options, key="selected_day")

min_date = df["date"].min().date()
max_date = df["date"].max().date()
start_date, end_date = st.sidebar.date_input(
    "Select date range:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    key="date_range"
)

st.sidebar.markdown("### Daily Open Position") 
adr_open_to__open = st.sidebar.selectbox("ADR Open Relative to Daily Open",
                                             ["All"] + sorted(df["adr_open_to_1800_open"].dropna().unique()),
                                             key="adr_open_to_1800_open_filter")
odr_open_to_1800_open = st.sidebar.selectbox("ODR Open Relative to Daily Open",
                                             ["All"] + sorted(df["odr_open_to_1800_open"].dropna().unique()),
                                             key="odr_open_to_1800_open_filter")
rdr_open_to_1800_open = st.sidebar.selectbox("RDR Open Relative to Daily Open",
                                             ["All"] + sorted(df["rdr_open_to_1800_open"].dropna().unique()),
                                             key="rdr_open_to_1800_open_filter")

st.sidebar.markdown("### Previous 15:55 Close Position") 
adr_open_to_prdr_1555_close = st.sidebar.selectbox("ADR Open Relative to PRDR 15:55 Close",
                                             ["All"] + sorted(df["adr_open_to_prev_1555_close"].dropna().unique()),
                                             key="adr_open_to_prdr_1555_close_filter")
odr_open_to_prdr_1555_close = st.sidebar.selectbox("ODR Open Relative to PRDR 15:55 Close",
                                             ["All"] + sorted(df["odr_open_to_prev_1555_close"].dropna().unique()),
                                             key="odr_open_to_prdr_1555_close_filter")
rdr_open_to_prdr_1555_close = st.sidebar.selectbox("RDR Open Relative to PRDR 15:55 Close",
                                             ["All"] + sorted(df["rdr_open_to_prev_1555_close"].dropna().unique()),
                                             key="rdr_open_to_prdr_1555_close_filter")


#########################################
### Resets
#########################################
default_filters = {
    "selected_day":                       "All",
    "date_range":                 (min_date, max_date),

    "prdr_mid_hit_filter":                "All",
    "adr_mid_hit_filter":                 "All",
    "odr_mid_hit_filter":                 "All",

    "prdr_mid_hit_filter_exclusion":      "None",
    "adr_mid_hit_filter_exclusion":       "None",
    "odr_mid_hit_filter_exclusion":       "None",

    "prdr_to_adr_model_filter" : [],
    "adr_to_rdr_model_filter" : [],
    "adr_to_odr_model_filter" : [],
    "odr_to_rdr_model_filter" : [],
    
    "prdr_conf_direction_filter" : "All",
    "adr_conf_direction_filter" : "All",
    "odr_conf_direction_filter" : "All",
    "rdr_conf_direction_filter" : "All",
    
    "prdr_conf_valid_filter" : "All",
    "adr_conf_valid_filter" : "All",
    "odr_conf_valid_filter" : "All",
    "rdr_conf_valid_filter" : "All",

    "adr_open_to_1800_open_filter" : "All",
    "odr_open_to_1800_open_filter" : "All",
    "rdr_open_to_1800_open_filter" : "All",

    "prev_rdr_box_color_filter" : "All",
    "adr_box_color_filter" : "All",
    "odr_box_color_filter" : "All",
    "rdr_box_color_filter" : "All",
    
    "adr_open_to_prdr_1555_close_filter" : "All",
    "odr_open_to_prdr_1555_close_filter" : "All",
    "rdr_open_to_prdr_1555_close_filter" : "All",

    "prdr_box_color_filter" : "All",
    "adr_box_color_filter" : "All",
    "odr_box_color_filter" : "All",
    "rdr_box_color_filter" : "All",
    
    "prdr_high_filter":                   "All", 
    "prdr_adr_transition_high_filter":    "All",
    "adr_high_filter":                    "All", 
    "adr_odr_transition_high_filter":     "All",
    "odr_high_filter":                    "All",
    "odr_rdr_transition_high_filter":     "All",
    "prdr_low_filter":                    "All",
    "prdr_adr_transition_low_filter":     "All", 
    "adr_low_filter":                     "All", 
    "adr_odr_transition_low_filter":      "All", 
    "odr_low_filter":                     "All", 
    "odr_rdr_transition_low_filter":      "All",

    "prdr_high_filter_exclusion":                   "None", 
    "prdr_adr_transition_high_filter_exclusion":    "None", 
    "adr_high_filter_exclusion":                    "None", 
    "adr_odr_transition_high_filter_exclusion":     "None", 
    "odr_high_filter_exclusion":                    "None", 
    "odr_rdr_transition_high_filter_exclusion":     "None", 
    "prdr_low_filter_exclusion":                    "None", 
    "prdr_adr_transition_low_filter_exclusion":     "None",  
    "adr_low_filter_exclusion":                     "None", 
    "adr_odr_transition_low_filter_exclusion":      "None",  
    "odr_low_filter_exclusion":                     "None", 
    "odr_rdr_transition_low_filter_exclusion":      "None", 
}

# 2) Reset button with callback
def reset_all_filters():
    for key, default in default_filters.items():
        # only touch keys that actually exist
        if key in st.session_state:
            st.session_state[key] = default

st.sidebar.button("Reset all filters", on_click=reset_all_filters)

if isinstance(start_date, tuple):
    # sometimes date_input returns a single date if you pass a single default
    start_date, end_date = start_date

st.markdown("### Dropdown Filters")

segment_order = list(segments.keys())          # ["pre_adr","adr","adr_transition",…,"rdr"]
segment_order_with_no = segment_order + ["Untouched"]

#########################################
### Session HoS / LoS Filters Inclusions
#########################################

with st.expander("Session High / Low Inclusions", expanded=False):
    row1_cols = st.columns([1, 1, 1, 1, 1, 1])
    row2_cols = st.columns([1, 1, 1, 1, 1, 1])
    
    with row1_cols[0]:
        prev_rdr_high_filter = st.selectbox(
            "PRDR High Touch",
            options=["All"] + ["Daily Open-ADR Transition", "ADR", "ADR-ODR Transition", "ODR", "ODR-RDR Transition", "RDR"],
            key="prdr_high_filter"
        )
    with row1_cols[1]:
        pre_adr_high_filter = st.selectbox(
            "Daily Open-ADR Transition High Touch",
            options=["All"] + ["ADR", "ADR-ODR Transition", "ODR", "ODR-RDR Transition", "RDR"],
            key="prdr_adr_transition_high_filter"
        )
    with row1_cols[2]:
        adr_high_filter = st.selectbox(
            "ADR High Touch",
            options=["All"] + ["ADR-ODR Transition", "ODR", "ODR-RDR Transition", "RDR"],
            key="adr_high_filter"
        )
    with row1_cols[3]:
        adr_transition_high_filter = st.selectbox(
            "ADR-ODR Transition High Touch",
            options=["All"] + ["ODR", "ODR-RDR Transition", "RDR"],
            key="adr_odr_transition_high_filter"
        )
    with row1_cols[4]:
        odr_high_filter = st.selectbox(
            "ODR High Touch",
            options=["All"] + ["ODR-RDR Transition", "RDR"],
            key="odr_high_filter"
        )
    with row1_cols[5]:
        odr_transition_high_filter = st.selectbox(
            "ODR-RDR Transition High Touch",
            options=["All"] + ["RDR"],
            key="odr_rdr_transition_high_filter"
        )

    # Second Row
    with row2_cols[0]:
        prev_rdr_low_filter = st.selectbox(
            "PRDR Low Touch",
            options=["All"] + ["Daily Open-ADR Transition", "ADR", "ADR-ODR Transition", "ODR", "ODR-RDR Transition", "RDR"],
            key="prdr_low_filter"
        )
    with row2_cols[1]:
        pre_adr_low_filter = st.selectbox(
            "Daily Open-ADR Transition Low Touch",
            options=["All"] + ["ADR", "ADR-ODR Transition", "ODR", "ODR-RDR Transition", "RDR"],
            key="prdr_adr_transition_low_filter"
        )
    with row2_cols[2]:
        adr_low_filter = st.selectbox(
            "ADR Low Touch",
            options=["All"] + ["ADR-ODR Transition", "ODR", "ODR-RDR Transition", "RDR"],
            key="adr_low_filter"
        )
    with row2_cols[3]:
        adr_transition_low_filter = st.selectbox(
            "ADR-ODR Transition Low Touch",
            options=["All"] + ["ODR", "ODR-RDR Transition", "RDR"],
            key="adr_odr_transition_low_filter"
        )
    with row2_cols[4]:
        odr_low_filter = st.selectbox(
            "ODR Low Touch",
            options=["All"] + ["ODR-RDR Transition", "RDR"],
            key="odr_low_filter"
        )
    with row2_cols[5]:
        odr_transition_low_filter = st.selectbox(
            "ODR-RDR Transition Low Touch",
            options=["All"] + ["RDR"],
            key="odr_rdr_transition_low_filter"
        )

#########################################
### Session HoS / LoS Filters Exclusions
#########################################
with st.expander("Session High / Low Exclusions", expanded=False):
    row3_cols = st.columns([1, 1, 1, 1, 1, 1])
    row4_cols = st.columns([1, 1, 1, 1, 1, 1])
    
    with row3_cols[0]:
        prev_rdr_high_filter_exclusion = st.selectbox(
            "PRDR High Touch",
            options=["None"] + ["Daily Open-ADR Transition", "ADR", "ADR-ODR Transition", "ODR", "ODR-RDR Transition", "RDR"],
            key="prdr_high_filter_exclusion"
        ) 
    with row3_cols[1]:
        pre_adr_high_filter_exclusion = st.selectbox(
            "Daily Open-ADR Transition High Touch",
            options=["None"] + ["ADR", "ADR-ODR Transition", "ODR", "ODR-RDR Transition", "RDR"],
            key="prdr_adr_transition_high_filter_exclusion"
        )
    with row3_cols[2]:
        adr_high_filter_exclusion = st.selectbox(
            "ADR High Touch",
            options=["None"] + ["ADR-ODR Transition", "ODR", "ODR-RDR Transition", "RDR"],
            key="adr_high_filter_exclusion"
        )
    with row3_cols[3]:
        adr_transition_high_filter_exclusion = st.selectbox(
            "ADR-ODR Transition High Touch",
            options=["None"] + ["ODR", "ODR-RDR Transition", "RDR"],
            key="adr_odr_transition_high_filter_exclusion"
        )
    with row3_cols[4]:
        odr_high_filter_exclusion = st.selectbox(
            "ODR High Touch",
            options=["None"] + ["ODR-RDR Transition", "RDR"],
            key="odr_high_filter_exclusion"
        )
    with row3_cols[5]:
        odr_transition_high_filter_exclusion = st.selectbox(
            "ODR-RDR Transition High Touch",
            options=["None"] + ["RDR"],
            key="odr_rdr_transition_high_filter_exclusion"
        )

    # Second Row
    with row4_cols[0]:
        prev_rdr_low_filter_exclusion = st.selectbox(
            "PRDR Low Touch",
            options=["None"] + ["Daily Open-ADR Transition", "ADR", "ADR-ODR Transition", "ODR", "ODR-RDR Transition", "RDR"],
            key="prdr_low_filter_exclusion"
        )
    with row4_cols[1]:
        pre_adr_low_filter_exclusion = st.selectbox(
            "Daily Open-ADR Transition Low Touch",
            options=["None"] + ["ADR", "ADR-ODR Transition", "ODR", "ODR-RDR Transition", "RDR"],
            key="prdr_adr_transition_low_filter_exclusion"
        )
    with row4_cols[2]:
        adr_low_filter_exclusion = st.selectbox(
            "ADR Low Touch",
            options=["None"] + ["ADR-ODR Transition", "ODR", "ODR-RDR Transition", "RDR"],
            key="adr_low_filter_exclusion"
        )
    with row4_cols[3]:
        adr_transition_low_filter_exclusion = st.selectbox(
            "ADR-ODR Transition Low Touch",
            options=["None"] + ["ODR", "ODR-RDR Transition", "RDR"],
            key="adr_odr_transition_low_filter_exclusion"
        )
    with row4_cols[4]:
        odr_low_filter_exclusion = st.selectbox(
            "ODR Low Touch",
            options=["None"] + ["ODR-RDR Transition", "RDR"],
            key="odr_low_filter_exclusion"
        )
    with row4_cols[5]:
        odr_transition_low_filter_exclusion = st.selectbox(
            "ODR-RDR Transition Low Touch",
            options=["None"] + ["RDR"],
            key="odr_rdr_transition_low_filter_exclusion"
        )

#########################################
### Midline Inclusion Filters
#########################################
with st.expander("Session IDR Midline Inclusions", expanded=False):
    row5_cols = st.columns([1, 1, 1])
    
    with row5_cols[0]:
        prev_rdr_midline_hit = st.selectbox(
            "PRDR Mid Touch",
            options=["All"] + ["Daily Open-ADR Transition", "ADR", "ADR-ODR Transition", "ODR", "ODR-RDR Transition", "RDR"],
            key="prdr_mid_hit_filter"
        )
    with row5_cols[1]:
        adr_midline_hit = st.selectbox(
            "ADR Mid Touch",
            options=["All"] + ["ADR-ODR Transition", "ODR", "ODR-RDR Transition", "RDR"],
            key="adr_mid_hit_filter"
        )
    with row5_cols[2]:
        odr_midline_hit = st.selectbox(
            "ODR Mid Touch",
            options=["All"] + ["ODR-RDR Transition", "RDR"],
            key="odr_mid_hit_filter"
        )
#########################################
### Midline Exclusion Filters
#########################################   
with st.expander("Session IDR Midline Exclusions", expanded=False):
    row6_cols = st.columns([1, 1, 1])
    with row6_cols[0]:
        prev_rdr_midline_hit_exclusion = st.selectbox(
            "PRDR Mid Touch",
            options=["None"] + ["Daily Open-ADR Transition", "ADR", "ADR-ODR Transition", "ODR", "ODR-RDR Transition", "RDR"],
            key="prdr_mid_hit_filter_exclusion"
        )
    with row6_cols[1]:
        adr_midline_hit_exclusion = st.selectbox(
            "ADR Mid Touch",
            options=["None"] + ["ADR-ODR Transition", "ODR", "ODR-RDR Transition", "RDR"],
            key="adr_mid_hit_filter_exclusion"
        )
    with row6_cols[2]:
        odr_midline_hit_exclusion = st.selectbox(
            "ODR Mid Touch",
            options=["None"] + ["ODR-RDR Transition", "RDR"],
            key="odr_mid_hit_filter_exclusion"
        )

#########################################
### Model Filters
#########################################
with st.expander("Models", expanded=False):
    row7_cols = st.columns([1, 1, 1, 1])
    with row7_cols[0]:
        prev_rdr_to_adr_model_filter = st.multiselect(
            "PRDR-ADR Model",
            options=["UXP", "UX", "U", "DXP", "DX", "D", "RC", "RX"],
            key="prdr_to_adr_model_filter",
        )
        
    with row7_cols[1]:
        adr_to_odr_model_filter = st.multiselect(
            "ADR-ODR Model",
            options=["UXP", "UX", "U", "DXP", "DX", "D", "RC", "RX"],
            key="adr_to_odr_model_filter", 
        )

    with row7_cols[2]:
        adr_to_rdr_model_filter = st.multiselect(
            "ADR-RDR Model",
            options=["UXP", "UX", "U", "DXP", "DX", "D", "RC", "RX"],
            key="adr_to_rdr_model_filter",
        )
        
    with row7_cols[3]:
        odr_to_rdr_model_filter = st.multiselect(
            "ODR-RDR Model",
            options=["UXP", "UX", "U", "DXP", "DX", "D", "RC", "RX"],
            key="odr_to_rdr_model_filter",
        )
        
#########################################
### Confirmation Direction Filter
#########################################
with st.expander("Confirmation Direction", expanded=False):
    row8_cols = st.columns([1, 1, 1, 1])
    with row8_cols[0]:
        prdr_conf_direction_filter = st.selectbox(
            "PRDR Confirmation Direction",
            options=["All"] + sorted(df["prev_rdr_conf_direction"].dropna().unique()),
            key="prdr_conf_direction_filter",
        )
    with row8_cols[1]:
        adr_conf_direction_filter = st.selectbox(
            "ADR Confirmation Direction",
            options=["All"] + sorted(df["adr_conf_direction"].dropna().unique()),
            key="adr_conf_direction_filter",
        )
    with row8_cols[2]:
        odr_conf_direction_filter = st.selectbox(
            "ODR Confirmation Direction",
            options=["All"] + sorted(df["odr_conf_direction"].dropna().unique()),
            key="odr_conf_direction_filter", 
        )
    with row8_cols[3]:
        rdr_conf_direction_filter = st.selectbox(
            "RDR Confirmation Direction",
            options=["All"] + sorted(df["rdr_conf_direction"].dropna().unique()),
            key="rdr_conf_direction_filter",
        )
        
#########################################
### True / False Filters
#########################################
with st.expander("Confirmation True/False", expanded=False):
    row9_cols = st.columns([1, 1, 1, 1])
    with row9_cols[0]:
        prdr_conf_valid_filter = st.selectbox(
            "PRDR Confirmation Valid",
            options=["All"] + sorted(df["prev_rdr_conf_valid"].dropna().unique(), reverse=True),
            key="prdr_conf_valid_filter",
        )
    with row9_cols[1]:
        adr_conf_valid_filter = st.selectbox(
            "ADR Confirmation Valid",
            options=["All"] + sorted(df["adr_conf_valid"].dropna().unique(), reverse=True),
            key="adr_conf_valid_filter",
        )
    with row9_cols[2]:
        odr_conf_valid_filter = st.selectbox(
            "ODR Confirmation Valid",
            options=["All"] + sorted(df["odr_conf_valid"].dropna().unique(), reverse=True),
            key="odr_conf_valid_filter",
        )
    with row9_cols[3]:
        rdr_conf_valid_filter = st.selectbox(
            "RDR Confirmation Valid",
            options=["All"] + sorted(df["rdr_conf_valid"].dropna().unique(), reverse=True),
            key="rdr_conf_valid_filter",
        )
        
#########################################
### Box Color Filters
#########################################
with st.expander("Box Color", expanded=False):
    row10_cols = st.columns([1, 1, 1, 1])
    with row10_cols[0]:
        prdr_box_color_filter = st.selectbox(
            "PRDR Box Color",
            options=["All"] + sorted(df["prev_rdr_box_color"].dropna().unique(), reverse=True),
            key="prdr_box_color_filter",
        )
    with row10_cols[1]:
        adr_box_color_filter = st.selectbox(
            "ADR Box Color",
            options=["All"] + sorted(df["adr_box_color"].dropna().unique(), reverse=True),
            key="adr_box_color_filter",
        )
    with row10_cols[2]:
        odr_box_color_filter = st.selectbox(
            "ODR Box Color",
            options=["All"] + sorted(df["odr_box_color"].dropna().unique(), reverse=True),
            key="odr_box_color_filter",
        )
    with row10_cols[3]:
        rdr_box_color_filter = st.selectbox(
            "RDR Box Color",
            options=["All"] + sorted(df["rdr_box_color"].dropna().unique(), reverse=True),
            key="rdr_box_color_filter",
        )

#########################################
### Filter Mapping
#########################################   

# map each filter to its column
inclusion_map = {
    "prev_rdr_high_touch_time_bucket":       "prdr_high_filter",
    "pre_adr_high_touch_time_bucket":        "prdr_adr_transition_high_filter",
    "adr_high_touch_time_bucket":            "adr_high_filter",
    "adr_transition_high_touch_time_bucket": "adr_odr_transition_high_filter",
    "odr_high_touch_time_bucket":            "odr_high_filter",
    "odr_transition_high_touch_time_bucket": "odr_rdr_transition_high_filter",

    "prev_rdr_low_touch_time_bucket":        "prdr_low_filter",
    "pre_adr_low_touch_time_bucket":         "prdr_adr_transition_low_filter",
    "adr_low_touch_time_bucket":             "adr_low_filter",
    "adr_transition_low_touch_time_bucket":  "adr_odr_transition_low_filter",
    "odr_low_touch_time_bucket":             "odr_low_filter",
    "odr_transition_low_touch_time_bucket":  "odr_rdr_transition_low_filter",

    "prev_rdr_idr_midline_touch_time_bucket":   "prdr_mid_hit_filter",
    "adr_idr_midline_touch_time_bucket":       "adr_mid_hit_filter",
    "odr_idr_midline_touch_time_bucket":       "odr_mid_hit_filter",

    "prev_rdr_to_adr_model" : "prdr_to_adr_model_filter",
    "adr_to_odr_model" : "adr_to_odr_model_filter",
    "adr_to_rdr_model" : "adr_to_rdr_model_filter",
    "odr_to_rdr_model" : "odr_to_rdr_model_filter",

    "adr_open_to_1800_open" : "adr_open_to_1800_open_filter",
    "odr_open_to_1800_open" : "odr_open_to_1800_open_filter",
    "rdr_open_to_1800_open" : "rdr_open_to_1800_open_filter",

    "adr_open_to_prev_1555_close" : "adr_open_to_prdr_1555_close_filter",
    "odr_open_to_prev_1555_close" : "odr_open_to_prdr_1555_close_filter",
    "rdr_open_to_prev_1555_close" : "rdr_open_to_prdr_1555_close_filter",

    "prev_rdr_box_color" : "prdr_box_color_filter",
    "adr_box_color" : "adr_box_color_filter",
    "odr_box_color" : "odr_box_color_filter",
    "rdr_box_color" : "rdr_box_color_filter",

    "prev_rdr_conf_direction" : "prdr_conf_direction_filter",
    "adr_conf_direction" : "adr_conf_direction_filter",
    "odr_conf_direction" : "odr_conf_direction_filter",
    "rdr_conf_direction" : "rdr_conf_direction_filter",

    "prev_rdr_conf_valid" : "prdr_conf_valid_filter",
    "adr_conf_valid" : "adr_conf_valid_filter",
    "odr_conf_valid" : "odr_conf_valid_filter",
    "rdr_conf_valid" : "rdr_conf_valid_filter",
}

exclusion_map = {
    "prev_rdr_high_touch_time_bucket":       "prdr_high_filter_exclusion",
    "pre_adr_high_touch_time_bucket":        "prdr_adr_transition_high_filter_exclusion",
    "adr_high_touch_time_bucket":            "adr_high_filter_exclusion",
    "adr_transition_high_touch_time_bucket": "adr_odr_transition_high_filter_exclusion",
    "odr_high_touch_time_bucket":            "odr_high_filter_exclusion",
    "odr_transition_high_touch_time_bucket": "odr_rdr_transition_high_filter_exclusion",

    "prev_rdr_low_touch_time_bucket":        "prdr_low_filter_exclusion",
    "pre_adr_low_touch_time_bucket":         "prdr_adr_transition_low_filter_exclusion",
    "adr_low_touch_time_bucket":             "adr_low_filter_exclusion",
    "adr_transition_low_touch_time_bucket":  "adr_odr_transition_low_filter_exclusion",
    "odr_low_touch_time_bucket":             "odr_low_filter_exclusion",
    "odr_transition_low_touch_time_bucket":  "odr_rdr_transition_low_filter_exclusion",

    "prev_rdr_idr_midline_touch_time_bucket":       "prdr_mid_hit_filter_exclusion",
    "adr_idr_midline_touch_time_bucket":            "adr_mid_hit_filter_exclusion",
    "odr_idr_midline_touch_time_bucket":            "odr_mid_hit_filter_exclusion",
}


# Apply filters
df_filtered = df.copy()

sel_day = st.session_state["selected_day"]
if sel_day != "All":
    df_filtered = df_filtered[df_filtered["day_of_week"]  == sel_day]

# — Date range —
start_date, end_date = st.session_state["date_range"]
df_filtered = df_filtered[
    (df_filtered["date"] >= pd.to_datetime(start_date)) &
    (df_filtered["date"] <= pd.to_datetime(end_date))
]

for col, state_key in inclusion_map.items():
    sel = st.session_state[state_key]
    if isinstance(sel, list):
        if sel:  # non-empty list means “only these”
            df_filtered = df_filtered[df_filtered[col].isin(sel)]
    else:
        if sel != "All":
            df_filtered = df_filtered[df_filtered[col] == sel]

#for col, state_key in exclusion_map.items():
    #excludes = st.session_state[state_key]
    #if excludes:
        #df_filtered = df_filtered[~df_filtered[col].isin(excludes)]

for col, state_key in exclusion_map.items():
    sel = st.session_state[state_key]   # now either "None" or a segment string
    if sel != "None":
        # build the full cascade from start up through 'sel'
        idx = segment_order.index(sel)
        to_exclude = set(segment_order[: idx+1])
        df_filtered = df_filtered[~df_filtered[col].isin(to_exclude)]
  
###########################################################
### True Rates, Box Color, and Conf. Direction Graphs
###########################################################
# Box Color and Confirmation Direction
true_rate_cols   = ["adr_conf_valid", "odr_conf_valid", "rdr_conf_valid"]
true_rate_titles = ["ADR True Rate", "ODR True Rate", "RDR True Rate"]

box_color_cols   = ["adr_box_color", "odr_box_color", "rdr_box_color"]
box_color_titles = ["ADR Box Color", "ODR Box Color", "RDR Box Color"]

conf_direction_cols   = ["adr_conf_direction", "odr_conf_direction", "rdr_conf_direction"]
conf_direction_titles = [
    "ADR Conf. Direction",
    "ODR Conf. Direction",
    "RDR Conf. Direction",
]

# make one 6‐column container
plot_df = df_filtered.copy()

for col in true_rate_cols:
    plot_df[col] = plot_df[col].map({True: "True", False: "False"})

# color maps
box_color_map = {
    "Green":   "#2ecc71",
    "Red":     "#e74c3c",
    "Neutral": "#5d6d7e",
}
dir_color_map = {
    "Long":  "#2ecc71", 
    "Short": "#e74c3c",
    "None":  "#5d6d7e",
}

true_color_map = {
    "True":  "#2ecc71",
    "False": "#e74c3c",
}

# replace null/NaN with the string "None" for just those three cols
for col in conf_direction_cols:
    plot_df[col] = plot_df[col].fillna("None")
    
all_cols = st.columns(len(box_color_cols) + len(conf_direction_cols) + len(true_rate_cols))

# true rate donuts
for i, col in enumerate(true_rate_cols):
    fig = px.pie(
        plot_df,
        names=col,
        color=col,                        # tell px to color by that column
        color_discrete_map=true_color_map, # map labels → colors
        title=true_rate_titles[i],
        hole=0.5,
    )
    fig.update_traces(textinfo="percent+label", textposition="inside", showlegend=False)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    all_cols[i].plotly_chart(fig, use_container_width=True)

# box-color donuts
offset = len(true_rate_cols)
for i, col in enumerate(box_color_cols):
    fig = px.pie(
        plot_df,
        names=col,
        color=col,                        # tell px to color by that column
        color_discrete_map=box_color_map, # map labels → colors
        title=box_color_titles[i],
        hole=0.5,
    )
    fig.update_traces(textinfo="percent+label", textposition="inside", showlegend=False)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    all_cols[offset + i].plotly_chart(fig, use_container_width=True)

offset2 = len(box_color_cols) + len(true_rate_cols)
for j, col in enumerate(conf_direction_cols):
    fig = px.pie(
        plot_df,
        names=col,
        color=col,
        color_discrete_map=dir_color_map,
        title=conf_direction_titles[j],
        hole=0.5,
    )
    fig.update_traces(textinfo="percent+label", textposition="inside", showlegend=False)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    all_cols[offset2 + j].plotly_chart(fig, use_container_width=True)

#########################################################
### 18:00 Hits
#########################################################
open_1800_cols = [
    "open_1800_adr_touch_time_buckets_v2",
    "open_1800_odr_touch_time_buckets_v2",
    "open_1800_rdr_touch_time_buckets_v2",
]
open_1800_titles = [
    "18:00 Open Hit in ADR",
    "18:00 Open Hit in ODR",
    "18:00 Open Hit in RDR"
]
prdr_gap_cols = [
    "prev_close_1555_adr_touch_time_buckets_v2",
    "prev_close_1555_odr_touch_time_buckets_v2",
    "prev_close_1555_rdr_touch_time_buckets_v2",
]
prdr_gap_titles = [
    "Prev. Day Gap Close in ADR",
    "Prev. Day Gap Close in ODR",
    "Prev. Day Gap Close in RDR"
]

open_1800_and_gap_row = st.columns(len(open_1800_cols) + len(prdr_gap_cols))

order = ["Box Formation", "Before Confirmation", "After Confirmation", "Untouched"]

for idx, col in enumerate(open_1800_cols):
    # 1) drop any actual None/NaT values
    series = df_filtered[col].fillna("Untouched")

    # 2) normalized counts, *then* reindex into your three‐bucket order
    counts = (
        series
        .value_counts(normalize=True)
        .reindex(order, fill_value=0)
    )

    # 4) turn into percentages
    perc = counts * 100
    perc = perc[perc > 0]

    # now build the bar‐chart
    fig = px.bar(
        x=perc.index,
        y=perc.values,
        text=[f"{v:.1f}%" for v in perc.values],
        title=open_1800_titles[idx],
        labels={"x": "", "y": ""},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_tickangle=90,
        margin=dict(l=10,r=10,t=30,b=10),
        yaxis=dict(showticklabels=False))

    open_1800_and_gap_row[idx].plotly_chart(fig, use_container_width=True)

for idx, col in enumerate(prdr_gap_cols):
    # 1) drop any actual None/NaN values so they never even show up
    series = df_filtered[col].fillna("Untouched")

    # 2) normalized counts, *then* reindex into your three‐bucket order
    counts = (
        series
        .value_counts(normalize=True)
        .reindex(order, fill_value=0)
    )

    # 4) turn into percentages
    perc = counts * 100
    perc = perc[perc > 0]

    # now build the bar‐chart
    fig = px.bar(
        x=perc.index,
        y=perc.values,
        text=[f"{v:.1f}%" for v in perc.values],
        title=prdr_gap_titles[idx],
        labels={"x": "", "y": ""},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_tickangle=90,
        margin=dict(l=10,r=10,t=30,b=10),
        yaxis=dict(showticklabels=False))

    open_1800_and_gap_row[idx+3].plotly_chart(fig, use_container_width=True)

#########################################################
### Models Graphs
#########################################################
model_cols = [
    "prev_rdr_to_adr_model",
    "adr_to_odr_model",
    "adr_to_rdr_model",
    "odr_to_rdr_model"]

model_titles = [
    "PRDR-ADR Model",
    "ADR-ODR Model",
    "ADR-RDR Model",
    "ODR-RDR Model"]

row1 = st.columns(len(model_cols))
for idx, col in enumerate(model_cols):
    # 1) drop any actual None/NaN values so they never even show up
    series = df_filtered[col].dropna() 

    # 2) get normalized counts
    counts = series.value_counts(normalize=True)

    # 3) if you still have the string "None" in your index, drop it
    counts = counts.drop("None", errors="ignore")

    # 4) turn into percentages
    perc = counts * 100
    perc = perc[perc > 0]

    # now build the bar‐chart
    fig = px.bar(
        x=perc.index,
        y=perc.values,
        text=[f"{v:.1f}%" for v in perc.values],
        title=model_titles[idx],
        labels={"x": "", "y": ""},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_tickangle=0,
        margin=dict(l=10,r=10,t=30,b=10),
        yaxis=dict(showticklabels=False))

    row1[idx].plotly_chart(fig, use_container_width=True)

#########################################################
### Midline Hits
#########################################################
mid_cols = [
    "prev_rdr_idr_midline_touch_time_bucket",
    "adr_idr_midline_touch_time_bucket",
    "odr_idr_midline_touch_time_bucket",
]
mid_titles = [
    "PRDR Mid",
    "ADR Mid",
    "ODR Mid",
]

row2 = st.columns(3)
for idx, col in enumerate(mid_cols):
    if col in df_filtered:
        counts = (
            df_filtered[col]
            .value_counts(normalize=True) 
            .reindex(segment_order_with_no, fill_value=0)
        )
        perc = counts * 100
        perc = perc[perc > 0]
        
        fig = px.bar(
            x=perc.index,
            y=perc.values,
            text=[f"{v:.1f}%" for v in perc.values],
            labels={"x": "", "y": ""},
            title=mid_titles[idx],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis_tickangle=90,
            yaxis=dict(showticklabels=False),
            xaxis={"categoryorder": "array", "categoryarray": list(perc.index)},
            margin=dict(l=10, r=10, t=30, b=10),
        )

        row2[idx].plotly_chart(fig, use_container_width=True)

#########################################################
### M7Box Hits
#########################################################
open_cols = [
    "prev_rdr_open_touch_time_bucket",
    "adr_open_touch_time_bucket",
    "odr_open_touch_time_bucket",

]
open_titles = [
    "PRDR Open Price",
    "ADR Open Price",
    "ODR Open Price",
]

open_close_row = st.columns(6)
for idx, col in enumerate(open_cols):
    if col in df_filtered:
        counts = (
            df_filtered[col]
            .value_counts(normalize=True)
            .reindex(segment_order_with_no, fill_value=0)
        )
        perc = counts * 100
        perc = perc[perc > 0]

        fig = px.bar(
            x=perc.index,
            y=perc.values,
            text=[f"{v:.1f}%" for v in perc.values],
            labels={"x": "", "y": ""},
            title=open_titles[idx],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis_tickangle=90,
            yaxis=dict(showticklabels=False),
            xaxis={"categoryorder": "array", "categoryarray": list(perc.index)},
            margin=dict(l=10, r=10, t=30, b=10),
        )

        open_close_row[idx].plotly_chart(fig, use_container_width=True)

close_cols = [
    "prev_rdr_close_touch_time_bucket",
    "adr_close_touch_time_bucket",
    "odr_close_touch_time_bucket",

]
close_titles = [
    "PRDR Close Price",
    "ADR Close Price",
    "ODR Close Price",
]

for idx, col in enumerate(close_cols):
    if col in df_filtered:
        counts = (
            df_filtered[col]
            .value_counts(normalize=True)
            .reindex(segment_order_with_no, fill_value=0)
        )
        perc = counts * 100
        perc = perc[perc > 0]

        fig = px.bar(
            x=perc.index,
            y=perc.values,
            text=[f"{v:.1f}%" for v in perc.values],
            labels={"x": "", "y": ""},
            title=close_titles[idx],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis_tickangle=90,
            yaxis=dict(showticklabels=False),
            xaxis={"categoryorder": "array", "categoryarray": list(perc.index)},
            margin=dict(l=10, r=10, t=30, b=10),
        )

        open_close_row[idx+3].plotly_chart(fig, use_container_width=True)

#########################################################
### Session High Hits
#########################################################
high_cols = [
    "prev_rdr_high_touch_time_bucket",
    "pre_adr_high_touch_time_bucket",
    "adr_high_touch_time_bucket",
    "adr_transition_high_touch_time_bucket",
    "odr_high_touch_time_bucket",
    "odr_transition_high_touch_time_bucket",
]
high_titles = [
    "PRDR High",
    "Daily Open-ADR Transition High",
    "ADR High",
    "ADR-ODR Transition High",
    "ODR High",
    "ODR-RDR Transition High",
]

row3 = st.columns(6)
for idx, col in enumerate(high_cols):
    if col in df_filtered:
        counts = (
            df_filtered[col]
            .value_counts(normalize=True)
            .reindex(segment_order_with_no, fill_value=0)
        )
        perc = counts * 100
        perc = perc[perc > 0]

        fig = px.bar(
            x=perc.index,
            y=perc.values,
            text=[f"{v:.1f}%" for v in perc.values],
            labels={"x": "", "y": ""},
            title=high_titles[idx],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis_tickangle=90,
            yaxis=dict(showticklabels=False),
            xaxis={"categoryorder": "array", "categoryarray": list(perc.index)},
            margin=dict(l=10, r=10, t=30, b=10),
        )

        row3[idx].plotly_chart(fig, use_container_width=True)


#########################################################
### Session Low Hits
#########################################################
low_cols = [
    "prev_rdr_low_touch_time_bucket",
    "pre_adr_low_touch_time_bucket",
    "adr_low_touch_time_bucket",
    "adr_transition_low_touch_time_bucket",
    "odr_low_touch_time_bucket",
    "odr_transition_low_touch_time_bucket",
]
low_titles = [
    "PRDR Low",
    "Daily Open-ADR Transition Low",
    "ADR Low",
    "ADR-ODR Transition Low",
    "ODR Low",
    "ODR-RDR Transition Low",
]

row4 = st.columns(6)
for idx, col in enumerate(low_cols):
    if col in df_filtered:
        counts = (
            df_filtered[col]
            .value_counts(normalize=True)
            .reindex(segment_order_with_no, fill_value=0)
        )
        perc = counts * 100
        perc = perc[perc > 0]

        fig = px.bar(
            x=perc.index,
            y=perc.values,
            text=[f"{v:.1f}%" for v in perc.values],
            labels={"x": "", "y": ""},
            title=low_titles[idx],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis_tickangle=90,
            yaxis=dict(showticklabels=False),
            xaxis={"categoryorder": "array", "categoryarray": list(perc.index)},
            margin=dict(l=10, r=10, t=30, b=10),
        )

        row4[idx].plotly_chart(fig, use_container_width=True)

#########################################################
### HoD / LoD Session Buckets
#########################################################
hod_lod_cols = [
    "hod",
    "lod",
]
hod_lod_titles = [
    "High of Day",
    "Low of Day"
]

hod_lod_row = st.columns(2)
for idx, col in enumerate(hod_lod_cols):
    if col in df_filtered:
        counts = (
            df_filtered[col]
            .value_counts(normalize=True)
            .reindex(segment_order_with_no, fill_value=0)
        )
        perc = counts * 100
        perc = perc[perc > 0]

        fig = px.bar(
            x=perc.index,
            y=perc.values,
            text=[f"{v:.1f}%" for v in perc.values],
            labels={"x": "", "y": ""},
            title=hod_lod_titles[idx],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis_tickangle=90,
            yaxis=dict(showticklabels=False),
            xaxis={"categoryorder": "array", "categoryarray": list(perc.index)},
            margin=dict(l=10, r=10, t=30, b=10),
        )

        hod_lod_row[idx].plotly_chart(fig, use_container_width=True)

#########################################################
### HoD / LoD 5m Buckets
#########################################################
times = [f"{h:02d}:{m:02d}" 
         for h in range(24) 
         for m in range(0, 60, 5)]

first_half  = [t for t in times if "18:00" <= t <= "23:55"]
second_half = [t for t in times if "00:00" <= t <= "15:55"]
rotated = first_half + second_half

hod_hm_cols = [
    "hod_hm",
]
hod_hm_titles = [
    "High of Day (5m)",
]

hod_row_hm = st.columns(len(hod_hm_cols))
for idx, col in enumerate(hod_hm_cols):
    if col in df_filtered:
        counts = (
            df_filtered[col]
            .value_counts(normalize=True)
        )
        perc = counts * 100
        perc = perc[perc > 0]

        y_vals = [perc.get(t, 0) for t in rotated]
        txt    = [f"{v:.1f}%"    for v in y_vals]

        fig = px.bar(
            x=rotated,
            y=y_vals,
            text=txt,
            labels={"x": "", "y": ""},
            title=hod_hm_titles[idx],
        )
        fig.update_traces(textposition="outside")
        fig.update_xaxes(categoryorder="array",
                        categoryarray=rotated,
                        tickangle=90)
        fig.update_layout(
            yaxis=dict(showticklabels=False),
            margin=dict(l=10, r=10, t=30, b=10),
        )

        hod_row_hm[idx].plotly_chart(fig, use_container_width=True)

lod_hm_cols = [
    "lod_hm",
]
lod_hm_titles = [
    "Low of Day (5m)",
]

lod_row_hm = st.columns(len(lod_hm_cols))
for idx, col in enumerate(lod_hm_cols):
    if col in df_filtered:
        counts = (
            df_filtered[col]
            .value_counts(normalize=True)
        )
        perc = counts * 100
        perc = perc[perc > 0]

        y_vals = [perc.get(t, 0) for t in rotated]
        txt    = [f"{v:.1f}%"    for v in y_vals]

        fig = px.bar(
            x=rotated,
            y=y_vals,
            text=txt,
            labels={"x": "", "y": ""},
            title=lod_hm_titles[idx],
        )
        fig.update_traces(textposition="outside")
        fig.update_xaxes(categoryorder="array",
                        categoryarray=rotated,
                        tickangle=90)
        fig.update_layout(
            yaxis=dict(showticklabels=False),
            margin=dict(l=10, r=10, t=30, b=10),
        )

        lod_row_hm[idx].plotly_chart(fig, use_container_width=True)

st.caption(f"Sample size: {len(df_filtered):,} rows")
