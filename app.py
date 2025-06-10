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

def load_data_for_instrument(instrument):
    df = pd.read_csv(f"https://raw.githubusercontent.com/TuckerArrants/adr_breakouts/refs/heads/main/{instrument}_ADR_Breakouts_From_2008.csv")
    return df

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

df['date'] = pd.to_datetime(df['date']).dt.date

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
    df["date"] = pd.to_datetime(df["date"])
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
### Breakout Times
#########################################

sessions = ['ADR', 'ADR-ODR Transition', 'ODR', 'ODR-RDR Transition', 'RDR']

with st.expander("Breakout Filters", expanded=True):
    row1_cols = st.columns([1, 1, 1, 1, 1, 1])
    row2_cols = st.columns([1, 1, 1, 1, 1, 1])
    
    with row1_cols[0]:
        breakout1_time_filter = st.selectbox(
            "1st Breakout Time",
            options=["None"] + sessions,
            key="breakout1_time_filter"
        )
    with row1_cols[1]:
        breakout2_time_filter = st.selectbox(
            "2nd Breakout Time",
            options=["None"] + sessions,
            key="breakout2_time_filter"
        )
    with row1_cols[2]:
        breakout3_time_filter = st.selectbox(
            "3rd Breakout Time",
            options=["None"] + sessions,
            key="breakout3_time_filter"
        )
    with row1_cols[3]:
        breakout4_time_filter = st.selectbox(
            "4th Breakout Time",
            options=["None"] + sessions,
            key="breakout4_time_filter"
        )
    with row1_cols[4]:
        breakout5_time_filter = st.selectbox(
            "5th Breakout Time",
            options=["None"] + sessions,
            key="breakout5_time_filter"
        )
    with row1_cols[5]:
        breakout6_time_filter = st.selectbox(
            "6th Breakout Time",
            options=["None"] + sessions,
            key="breakout6_time_filter"
        )

#########################################
### Filter Mapping
#########################################   

# map each filter to its column
inclusion_map = {
    "breakout_segment1": "breakout1_time_filter",
    "breakout_segment2": "breakout2_time_filter",
    "breakout_segment3": "breakout3_time_filter",
    "breakout_segment4": "breakout4_time_filter",
    "breakout_segment5": "breakout5_time_filter",
    "breakout_segment6": "breakout6_time_filter",
}

# Apply filters
df_filtered = df.copy()

sel_day = st.session_state["selected_day"]
#if sel_day != "All":
    #df_filtered = df_filtered[df_filtered["day_of_week"]  == sel_day]

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
        if sel != "None":
            df_filtered = df_filtered[df_filtered[col] == sel]

#########################################################
### Breakout Time Distributions
#########################################################

breakout_time_cols = [
    "breakout_time1",
    "breakout_time2",
    "breakout_time3",
    "breakout_time4",
    "breakout_time5",
    "breakout_time6",
]
order = [
    "ADR",
    "ADR-ODR Transition",
    "ODR",
    "ODR-RDR Transition",
    "RDR"
]
cols = st.columns(len(breakout_time_cols))

for idx, time_col in enumerate(breakout_time_cols):
    seg_col = f"breakout_segment{idx+1}"
    series = df_filtered[seg_col]

    # normalized counts in your exact order
    counts = (
        series
        .value_counts(normalize=True)
        .reindex(order, fill_value=0)
    )
    df_plot = counts.mul(100).reset_index()
    df_plot.columns = ['segment', 'percentage']
    df_plot['text'] = df_plot['percentage'].map(lambda v: f"{v:.1f}%")

    # Use "Breakout {n}" as the title 
    fig = px.bar(
        df_plot,
        x='segment',
        y='percentage',
        text='text',
        title=f"Breakout {idx+1}",
        labels={'segment': '', 'percentage': ''},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_tickangle=90,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(showticklabels=False)
    )

    cols[idx].plotly_chart(fig, use_container_width=True)

# ── Overall # of Breakouts Distribution ────────────────────────────────────────
MAX_N_BREAKOUTS=10
dir_cols      = [f'breakout_direction{i}' for i in range(1, MAX_N_BREAKOUTS+1)]
counts_per_day = df_filtered[dir_cols].notnull().sum(axis=1)

# 2) Build a Series indexed 0..MAX_N_BREAKOUTS with counts of days
freq = counts_per_day.value_counts().sort_index()
freq = freq.reindex(range(MAX_N_BREAKOUTS+1), fill_value=0)  # ensures 0 through 6 are present

# 3) Turn into a DataFrame
df_dist = freq.rename_axis('num_breakouts').reset_index(name='days')

# 4) Compute % text if you like
df_dist['pct_label'] = (
    df_dist['days'] / df_dist['days'].sum() * 100
).map(lambda v: f"{v:.1f}%")

# ── Plot with Plotly ───────────────────────────────────────────────────────────
fig = px.bar(
    df_dist,
    x='num_breakouts',
    y='days',
    text='pct_label',  # or 'days' if you prefer raw counts
    title='Distribution of Number of Breakouts per Day',
    labels={'num_breakouts': '# Breakouts', 'days': 'Number of Days'},
)

fig.update_traces(textposition='outside')
fig.update_layout(
    xaxis=dict(dtick=1),
    margin=dict(l=20, r=20, t=40, b=20),
)

st.plotly_chart(fig, use_container_width=True)
st.caption(f"Sample size: {len(df_filtered):,} rows")
