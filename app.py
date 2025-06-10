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
def bucket_times(df: pd.DataFrame, time_cols: list) -> pd.DataFrame:
    df = df.copy()

    def _segment_for_timestamp(ts):
        if pd.isna(ts):
            return None
        # ensure it's a Timestamp
        t = ts.time() if isinstance(ts, pd.Timestamp) else pd.to_datetime(ts).time()
        if time(19, 30) <= t <= time(23, 55):
            return 'adr'
        if time(0, 0) <= t < time(2, 0):
            return 'adr'
        if time(2, 0) <= t < time(3, 0):
            return 'adr_transition'
        if time(3, 0) <= t < time(8, 0):
            return 'odr'
        if time(8, 0) <= t < time(9, 30):
            return 'rdr_transition'
        if time(9, 30) <= t < time(16, 00):
            return 'rdr'
        return None

    for i, col in enumerate(time_cols, start=1):
        seg_col = f'breakout_segment{i}'
        df[seg_col] = df[col].apply(_segment_for_timestamp)

    return df

def load_data_for_instrument(instrument)
    df = pd.read_csv(f"https://raw.githubusercontent.com/TuckerArrants/adr_breakouts/refs/heads/main/{INSTRUMENT}_ADR_Breakouts_From_2008.csv")

# ✅ Store username-password pairs
USER_CREDENTIALS = {
    "badboyz": "bangbang",
    "dreamteam" : "strike",
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
st.title("ADR Breakouts")

# ↓ in your sidebar:
instrument_options = ["ES", "NQ", "YM", "CL", "GC"]
selected_instrument = st.sidebar.selectbox("Instrument", instrument_options)

#########################################
### Data Loading and Processing
#########################################
df = load_data_for_instrument(selected_instrument)
breakout_time_cols = [col for col in df.columns if col.startswith('breakout_time')]

df['date'] = pd.to_datetime(df['session_date']).dt.date
df = bucket_times(df, breakout_time_cols)

rename_map = {
              'adr' : 'ADR',
              'adr_transition' : 'ADR-ODR Transition',
              'odr' : 'ODR',
              'odr_transition' : 'ODR-RDR Transition',
              'rdr' : 'RDR',
              'untouched' : 'Untouched',
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


#########################################
### Resets
#########################################
default_filters = {
    "selected_day":                       "All",
    "date_range":                 (min_date, max_date),
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


#########################################
### Session HoS / LoS Filters Inclusions
#########################################

sessions = ['ADR', 'ADR Transition', 'ODR', 'ODR Transition', 'RDR']

with st.expander("Breakout Fitlers", expanded=True):
    row1_cols = st.columns([1, 1, 1, 1, 1, 1])
    row2_cols = st.columns([1, 1, 1, 1, 1, 1])
    
    with row1_cols[0]:
        breakout1_time_filter = st.selectbox(
            "1st Breakout Time",
            options=["All"] + sessions,
            key="breakout1_time_filter"
        )
    with row1_cols[1]:
        breakout2_time_filter = st.selectbox(
            "2nd Breakout Time",
            options=["All"] + sessions,
            key="breakout2_time_filter"
        )
    with row1_cols[2]:
        breakout3_time_filter = st.selectbox(
            "3rd Breakout Time",
            options=["All"] + sessions,
            key="breakout3_time_filter"
        )
    with row1_cols[3]:
        breakout4_time_filter = st.selectbox(
            "4th Breakout Time",
            options=["All"] + sessions,
            key="breakout4_time_filter"
        )
    with row1_cols[4]:
        breakout5_time_filter = st.selectbox(
            "5th Breakout Time",
            options=["All"] + sessions,
            key="breakout5_time_filter"
        )
    with row1_cols[5]:
        breakout6_time_filter = st.selectbox(
            "6th Breakout Time",
            options=["All"] + sessions,
            key="breakout6_time_filter"
        )

#########################################
### Filter Mapping
#########################################   

# map each filter to its column
inclusion_map = {

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

#########################################################
### Breakout Time Distributions
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


st.caption(f"Sample size: {len(df_filtered):,} rows")
