import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyproj import Geod
import io
import csv
import math

st.set_page_config(page_title="CCC Vizuální Analýza Pojezdu", layout="wide")
st.title("🗺️ CCC: Komplexní vizuální analýza stopy a zkoušky")
st.caption("Globální přehled stavby -> Lokální detail zkoušky -> Surová data pojezdu.")

# --- 1. PARSOVÁNÍ DAT ---
@st.cache_data
def nacti_surova_data(file_bytes):
    sample_text = file_bytes[:50000].decode("utf-8", errors="ignore")
    lines = sample_text.splitlines()
    header_idx = 0
    for i, line in enumerate(lines):
        if any(k in line.lower() for k in ["latitude", "lat", "time", "gps:"]):
            header_idx = i; break
    header_line = lines[header_idx]
    try: sep = csv.Sniffer().sniff(header_line).delimiter
    except: sep = ';' if header_line.count(';') > header_line.count(',') else ','
    
    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, skiprows=header_idx, on_bad_lines='skip', dtype=str, low_memory=False)
    df.columns = df.columns.astype(str).str.strip().str.replace('"', '').str.replace("'", "")
    return df

def najdi_sloupec(columns, klicova_slova):
    for col in columns:
        if any(slovo in str(col).lower() for slovo in klicova_slova): return col
    return columns[0] if len(columns) > 0 else None

# --- UI SIDEBAR ---
with st.sidebar:
    st.header("📂 Vstupní data")
    uploaded_file = st.file_uploader("Nahrát CSV z válce", type=['csv'])
    
    if uploaded_file:
        df_raw = nacti_surova_data(uploaded_file.getvalue())
        col_time = st.selectbox("Čas", df_raw.columns, index=df_raw.columns.get_loc(najdi_sloupec(df_raw.columns, ['time', 'cas'])))
        col_lat = st.selectbox("Latitude", df_raw.columns, index=df_raw.columns.get_loc(najdi_sloupec(df_raw.columns, ['lat'])))
        col_lon = st.selectbox("Longitude", df_raw.columns, index=df_raw.columns.get_loc(najdi_sloupec(df_raw.columns, ['lon'])))
        col_stiff = st.selectbox("Tuhost (Kb)", df_raw.columns, index=df_raw.columns.get_loc(najdi_sloupec(df_raw.columns, ['stiff', 'kb', 'cmv'])))
        col_vib = st.selectbox("Amplituda/Vibrace", [None] + list(df_raw.columns), index=0)
        
        st.header("🎯 Místo zkoušky")
        target_lat = st.number_input("Zkouška Y (Lat)", value=50.0791600, format="%.7f")
        target_lon = st.number_input("Zkouška X (Lon)", value=14.5930200, format="%.7f")
        
        st.header("📐 Stroj")
        offset_fwd = st.number_input("Podélný posun antény (m)", value=2.00)
        offset_right = st.number_input("Příčný posun antény (m)", value=0.00)
        roller_width = st.number_input("Šířka běhounu (m)", value=2.10)

# --- JÁDRO A VÝPOČTY ---
if uploaded_file is not None:
    # Konverze a čištění
    df = df_raw.copy()
    df['lat'] = pd.to_numeric(df[col_lat].astype(str).str.replace(',', '.'), errors='coerce')
    df['lon'] = pd.to_numeric(df[col_lon].astype(str).str.replace(',', '.'), errors='coerce')
    df['Kb'] = pd.to_numeric(df[col_stiff].astype(str).str.replace(',', '.'), errors='coerce')
    df['Amp'] = pd.to_numeric(df[col_vib].astype(str).str.replace(',', '.'), errors='coerce') if col_vib else np.nan
    df['parsed_time'] = pd.to_datetime(df[col_time].astype(str).str.replace(' GMT', ''), utc=True, format='mixed', errors='coerce')
    df = df.dropna(subset=['lat', 'lon', 'parsed_time', 'Kb']).sort_values('parsed_time').reset_index(drop=True)
    df['Bod_ID'] = df.index + 1

    geod = Geod(ellps="WGS84")

    # KINEMATIKA: Výpočet azimutu a středu válce
    df['smooth_lon'] = df['lon'].rolling(3, min_periods=1, center=True).mean()
    df['smooth_lat'] = df['lat'].rolling(3, min_periods=1, center=True).mean()
    fwd_az, _, _ = geod.inv(df['smooth_lon'].shift().bfill().values, df['smooth_lat'].shift().bfill().values, df['smooth_lon'].values, df['smooth_lat'].values)
    df['heading'] = fwd_az % 360
    
    # Promítnutí offsetů na střed běhounu
    df['mid_lon'], df['mid_lat'], _ = geod.fwd(df['lon'].values, df['lat'].values, df['heading'].values, np.full(len(df), offset_fwd))
    h_right = (df['heading'] - 90) % 360 if offset_right < 0 else (df['heading'] + 90) % 360
    df['drum_lon'], df['drum_lat'], _ = geod.fwd(df['mid_lon'].values, df['mid_lat'].values, h_right, np.full(len(df), abs(offset_right)))

    # NALEZENÍ ZÁSAHU: Který bod je nejblíže zkoušce?
    _, _, dists = geod.inv(df['drum_lon'].values, df['drum_lat'].values, np.full(len(df), target_lon), np.full(len(df), target_lat))
    df['dist_to_test_m'] = dists
    idx_closest = df['dist_to_test_m'].idxmin()
    cpa_time = df.loc[idx_closest, 'parsed_time']
    
    # EXTRAKCE POJEZDU: Vezmeme plynulý pojezd (např. 15 vteřin před a po nejbližším bodě)
    df_pass = df[
        (df['parsed_time'] >= cpa_time - pd.Timedelta(seconds=15)) & 
        (df['parsed_time'] <= cpa_time + pd.Timedelta(seconds=15))
    ].copy()

    # VÝPOČET GEOMETRIE PÁSU PRO TENTO POJEZD
    hl_lon, hl_lat, hr_lon, hr_lat = [], [], [], []
    for _, row in df_pass.iterrows():
        l_lon, l_lat, _ = geod.fwd(row['drum_lon'], row['drum_lat'], (row['heading'] - 90) % 360, roller_width / 2)
        r_lon, r_lat, _ = geod.fwd(row['drum_lon'], row['drum_lat'], (row['heading'] + 90) % 360, roller_width / 2)
        hl_lon.append(l_lon); hl_lat.append(l_lat)
        hr_lon.append(r_lon); hr_lat.append(r_lat)
    
    poly_lon = hl_lon + hr_lon[::-1] + [hl_lon[0]]
    poly_lat = hl_lat + hr_lat[::-1] + [hl_lat[0]]

    # Odsazení a status
    az_to_target, _, _ = geod.inv(df.loc[idx_closest, 'drum_lon'], df.loc[idx_closest, 'drum_lat'], target_lon, target_lat)
    cross_track = abs(df.loc[idx_closest, 'dist_to_test_m'] * math.sin(math.radians(az_to_target - df.loc[idx_closest, 'heading'])))
    
    if cross_track <= (roller_width / 2):
        st.success(f"🎯 Zkouška leží v pásu stroje. Přesná příčná odchylka osy: **{cross_track:.2f} m**.")
    else:
        st.warning(f"⚠️ Zkouška leží MIMO hlavní pás (Příčná odchylka osy: {cross_track:.2f} m, šířka polovičního běhounu je {roller_width/2:.2f} m).")

    # --- UI LAYOUT ---
    tab1, tab2 = st.tabs(["🌍 1. Globální Mapa & Historie", "🔍 2. Detailní Přiblížení a Data"])
    cos_corr = 1 / np.cos(np.radians(target_lat))

    with tab1:
        st.subheader("Všechny pojezdy a identifikace zkoušky")
        fig_global = go.Figure()
        
        # 1. Všechny raw body stroje (Slabě)
        fig_global.add_trace(go.Scattergl(
            x=df['drum_lon'], y=df['drum_lat'], mode='lines', 
            line=dict(color='lightgrey', width=1), name='Ostatní pojezdy'
        ))
        
        # 2. Polygon identifikovaného pásu
        fig_global.add_trace(go.Scatter(
            x=poly_lon, y=poly_lat, mode='lines', fill='toself',
            fillcolor='rgba(0, 150, 136, 0.4)', line=dict(color='teal', width=1.5), name=f'Zásah: Pás {roller_width}m'
        ))
        
        # 3. Osa vybraného pojezdu
        fig_global.add_trace(go.Scatter(
            x=df_pass['drum_lon'], y=df_pass['drum_lat'], mode='lines', 
            line=dict(color='teal', width=3), name='Osa zásahu'
        ))

        # 4. Zkouška
        fig_global.add_trace(go.Scatter(
            x=[target_lon], y=[target_lat], mode='markers',
            marker=dict(size=14, symbol='x', color='red', line=dict(width=3)), name='Místo Zkoušky'
        ))

        fig_global.update_layout(
            yaxis=dict(scaleanchor="x", scaleratio=cos_corr),
            height=600, margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_global, use_container_width=True)

    with tab2:
        col_map, col_data = st.columns([1.5, 1])
        
        with col_map:
            st.subheader("Lokalizovaný detail")
            fig_local = go.Figure()
            
            # Polygon pásu
            fig_local.add_trace(go.Scatter(
                x=poly_lon, y=poly_lat, mode='lines', fill='toself',
                fillcolor='rgba(0, 150, 136, 0.2)', line=dict(color='teal', width=2, dash='dash'), name='Okraje běhounu'
            ))
            
            # Osa a body (s hodnotou Kb)
            fig_local.add_trace(go.Scatter(
                x=df_pass['drum_lon'], y=df_pass['drum_lat'], mode='lines+markers+text',
                marker=dict(size=10, color='black'), line=dict(color='teal', width=3),
                text=df_pass['Kb'], textposition="top center",
                name='Hodnoty Kb'
            ))
            
            # Zkouška
            fig_local.add_trace(go.Scatter(
                x=[target_lon], y=[target_lat], mode='markers',
                marker=dict(size=18, symbol='x', color='red', line=dict(width=3)), name='Zkouška'
            ))

            # Nastavení fixního zoomu kolem zkoušky (+- cca 10 metrů)
            buf = 0.0001
            fig_local.update_layout(
                yaxis=dict(scaleanchor="x", scaleratio=cos_corr, tickformat=".7f", range=[target_lat-buf, target_lat+buf]),
                xaxis=dict(tickformat=".7f", range=[target_lon-buf, target_lon+buf]),
                height=600, margin=dict(l=0, r=0, t=30, b=0), showlegend=False
            )
            st.plotly_chart(fig_local, use_container_width=True)

        with col_data:
            st.subheader("Historie bodů pojezdu")
            st.caption("Seřazeno chronologicky. Souřadnice na 7 desetinných míst.")
            
            df_table = df_pass[['Bod_ID', 'parsed_time', 'drum_lon', 'drum_lat', 'Kb', 'Amp', 'dist_to_test_m']].copy()
            df_table['parsed_time'] = df_table['parsed_time'].dt.strftime('%H:%M:%S')
            df_table['drum_lon'] = df_table['drum_lon'].map('{:.7f}'.format)
            df_table['drum_lat'] = df_table['drum_lat'].map('{:.7f}'.format)
            df_table['dist_to_test_m'] = df_table['dist_to_test_m'].round(2)
            df_table = df_table.rename(columns={'dist_to_test_m': 'Vzdál. od zk. (m)'})
            
            st.dataframe(df_table, hide_index=True, use_container_width=True, height=550)
else:
    st.info("Nahraj data.")
