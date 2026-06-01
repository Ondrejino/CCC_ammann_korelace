import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyproj import Geod
import io
import csv
import math

st.set_page_config(page_title="CCC Čisté Osové Párování", layout="wide")
st.title("🎯 CCC: Nalezení 2 nejbližších bodů ve stopě válce")

# --- PARSER ---
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

# --- UI ---
with st.sidebar:
    st.header("📂 Vstupní data")
    uploaded_file = st.file_uploader("Nahrát CSV z válce", type=['csv'])
    
    if uploaded_file:
        df_raw = nacti_surova_data(uploaded_file.getvalue())
        col_lat = st.selectbox("Latitude", df_raw.columns, index=df_raw.columns.get_loc(najdi_sloupec(df_raw.columns, ['lat'])))
        col_lon = st.selectbox("Longitude", df_raw.columns, index=df_raw.columns.get_loc(najdi_sloupec(df_raw.columns, ['lon'])))
        col_stiff = st.selectbox("Tuhost (Kb)", df_raw.columns, index=df_raw.columns.get_loc(najdi_sloupec(df_raw.columns, ['stiff', 'kb', 'cmv'])))
        
        st.header("🎯 Cíl (Zkouška)")
        target_lat = st.number_input("Zkouška Y (Lat)", value=50.0791600, format="%.7f")
        target_lon = st.number_input("Zkouška X (Lon)", value=14.5930200, format="%.7f")
        
        st.header("📐 Rozměry stroje")
        offset_fwd = st.number_input("Podélný posun (m)", value=2.00)
        roller_width = st.number_input("Šířka běhounu (m)", value=2.10)

# --- JÁDRO ---
if uploaded_file is not None:
    # 1. Příprava dat
    df = df_raw.copy()
    df['lat'] = pd.to_numeric(df[col_lat].astype(str).str.replace(',', '.'), errors='coerce')
    df['lon'] = pd.to_numeric(df[col_lon].astype(str).str.replace(',', '.'), errors='coerce')
    df['Kb'] = pd.to_numeric(df[col_stiff].astype(str).str.replace(',', '.'), errors='coerce')
    df = df.dropna(subset=['lat', 'lon', 'Kb']).reset_index(drop=True)
    df['Bod_ID'] = df.index + 1

    geod = Geod(ellps="WGS84")

    # Geometrie: Vypočítáme rovnou střed válce (zjednodušeně posun vpřed podle azimutu mezi body)
    fwd_az, _, _ = geod.inv(df['lon'].shift().bfill().values, df['lat'].shift().bfill().values, df['lon'].values, df['lat'].values)
    df['heading'] = fwd_az % 360
    df['drum_lon'], df['drum_lat'], _ = geod.fwd(df['lon'].values, df['lat'].values, df['heading'].values, np.full(len(df), offset_fwd))

    # 2. Hledání správného pojezdu a bodů
    # Spočítáme radiální vzdálenost všech bodů ke zkoušce
    _, _, dists_to_target = geod.inv(df['drum_lon'].values, df['drum_lat'].values, np.full(len(df), target_lon), np.full(len(df), target_lat))
    df['dist_m'] = dists_to_target

    # Vezmeme 100 absolutně nejbližších bodů, abychom zkontrolovali křížení
    kandidati = df.nsmallest(100, 'dist_m').sort_values('Bod_ID')
    
    # Najdeme absolutně nejbližší bod (Bod 1)
    if not kandidati.empty:
        idx_closest = kandidati['dist_m'].idxmin()
        bod_1 = df.loc[idx_closest]
        
        # Bod 2 musí být hned vedle Bodu 1 v čase/historii (+1 nebo -1 index)
        idx_prev = idx_closest - 1 if idx_closest > 0 else idx_closest
        idx_next = idx_closest + 1 if idx_closest < len(df) - 1 else idx_closest
        
        # Z předchozího a následujícího vybereme ten, který je blíž zkoušce
        if df.loc[idx_prev, 'dist_m'] < df.loc[idx_next, 'dist_m']:
            bod_2 = df.loc[idx_prev]
        else:
            bod_2 = df.loc[idx_next]

        # Vytvoříme finální tabulku 2 bodů
        df_final = pd.DataFrame([bod_1, bod_2]).sort_values('Bod_ID')
        
        # Kontrola, jestli válec zkoušku fakt přejel (příčná odchylka hrubým odhadem)
        az_to_target, _, _ = geod.inv(bod_1['drum_lon'], bod_1['drum_lat'], target_lon, target_lat)
        cross_track = abs(bod_1['dist_m'] * math.sin(math.radians(az_to_target - bod_1['heading'])))
        
        if cross_track > (roller_width / 2):
            st.warning(f"⚠️ Nejbližší nalezený pojezd je mimoběžný. Střed válce byl {cross_track:.2f} m od zkoušky (mimo běhoun).")
        
        # --- ZOBRAZENÍ VÝSLEDKŮ ---
        st.subheader("✅ Nalezená úsečka přes zkoušku")
        
        # Výpočet interpolovaného Kb
        avg_kb = df_final['Kb'].mean()
        st.success(f"**Výsledné průměrné Kb z těchto dvou bodů: {avg_kb:.1f}**")
        
        st.dataframe(df_final[['Bod_ID', 'Kb', 'dist_m', 'drum_lon', 'drum_lat']], hide_index=True)
        
        # --- VIZUALIZACE (Čistý obdélník 2 bodů) ---
        fig = go.Figure()

        # Vypočítáme 4 rohy běhounu POUZE pro tyto 2 body
        l1_lon, l1_lat, _ = geod.fwd(df_final.iloc[0]['drum_lon'], df_final.iloc[0]['drum_lat'], (df_final.iloc[0]['heading'] - 90) % 360, roller_width / 2)
        r1_lon, r1_lat, _ = geod.fwd(df_final.iloc[0]['drum_lon'], df_final.iloc[0]['drum_lat'], (df_final.iloc[0]['heading'] + 90) % 360, roller_width / 2)
        l2_lon, l2_lat, _ = geod.fwd(df_final.iloc[1]['drum_lon'], df_final.iloc[1]['drum_lat'], (df_final.iloc[1]['heading'] - 90) % 360, roller_width / 2)
        r2_lon, r2_lat, _ = geod.fwd(df_final.iloc[1]['drum_lon'], df_final.iloc[1]['drum_lat'], (df_final.iloc[1]['heading'] + 90) % 360, roller_width / 2)

        # Polygon pásu
        poly_lon = [l1_lon, r1_lon, r2_lon, l2_lon, l1_lon]
        poly_lat = [l1_lat, r1_lat, r2_lat, l2_lat, l1_lat]

        fig.add_trace(go.Scatter(
            x=poly_lon, y=poly_lat, mode='lines', fill='toself',
            fillcolor='rgba(0, 150, 136, 0.3)', line=dict(color='teal', width=2), name=f'Pás běhounu'
        ))

        # Osa válce
        fig.add_trace(go.Scatter(
            x=df_final['drum_lon'], y=df_final['drum_lat'], mode='lines+markers',
            marker=dict(size=12, color='black'), line=dict(color='black', width=3, dash='dot'),
            name='Osa (Spojnice bodů)'
        ))

        # Bod zkoušky
        fig.add_trace(go.Scatter(
            x=[target_lon], y=[target_lat], mode='markers',
            marker=dict(size=16, symbol='x', color='red', line=dict(width=3)), name='Zkouška'
        ))

        cos_corr = 1 / np.cos(np.radians(target_lat))
        # Odsazení pro čisté zobrazení
        buf = 0.00003
        fig.update_layout(
            yaxis=dict(scaleanchor="x", scaleratio=cos_corr, tickformat=".7f", range=[target_lat-buf, target_lat+buf]),
            xaxis=dict(tickformat=".7f", range=[target_lon-buf, target_lon+buf]),
            height=600, margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Nepodařilo se najít žádné body.")
