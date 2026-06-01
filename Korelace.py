import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyproj import Geod
import io
import csv
import math

st.set_page_config(page_title="CCC Manuální Osové Párování", layout="wide")
st.title("🎯 CCC: Expertní Manuální Párování ve Stopě Válce")
st.caption("Filtruje pouze body z konkrétního pojezdu, jehož běhoun (2.1 m) protnul zkoušku.")

# --- PARSER A GEODATA (Vychází z tvé v3.1) ---
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

@st.cache_data
def zpracuj_kinematiku(df_raw, col_lat, col_lon, col_time, col_speed, col_dir, offset_fwd, offset_right, min_speed, fwd_val):
    df = df_raw.copy()
    for col in [col_lat, col_lon]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    df['parsed_time'] = pd.to_datetime(df[col_time].astype(str).str.replace(' GMT', ''), utc=True, format='mixed', errors='coerce')
    df = df.dropna(subset=[col_lat, col_lon, 'parsed_time']).sort_values('parsed_time').reset_index(drop=True)
    
    geod = Geod(ellps="WGS84")
    df['smooth_lon'] = df[col_lon].rolling(3, min_periods=1, center=True).mean()
    df['smooth_lat'] = df[col_lat].rolling(3, min_periods=1, center=True).mean()
    
    fwd_az, _, _ = geod.inv(df['smooth_lon'].shift(2).bfill().values, df['smooth_lat'].shift(2).bfill().values, df['smooth_lon'].values, df['smooth_lat'].values)
    gps_heading = fwd_az % 360
    
    if col_dir in df.columns:
        is_fwd = (df[col_dir].astype(str).str.strip() == str(fwd_val)).values
        df['heading'] = np.where(is_fwd, gps_heading, (gps_heading + 180) % 360)
    else:
        df['heading'] = gps_heading

    # Geometrie
    df['mid_lon'], df['mid_lat'], _ = geod.fwd(df[col_lon].values, df[col_lat].values, df['heading'].values, np.full(len(df), offset_fwd))
    h_right = (df['heading'] - 90) % 360 if offset_right < 0 else (df['heading'] + 90) % 360
    df['drum_lon'], df['drum_lat'], _ = geod.fwd(df['mid_lon'].values, df['mid_lat'].values, h_right, np.full(len(df), abs(offset_right)))
    
    # Rychlost a Pojezdy
    df['dt'] = df['parsed_time'].diff().dt.total_seconds().replace(0, 0.01).bfill()
    _, _, dist = geod.inv(df['drum_lon'].shift().bfill().values, df['drum_lat'].shift().bfill().values, df['drum_lon'].values, df['drum_lat'].values)
    df['speed_kmh'] = (dist / df['dt']) * 3.6 if col_speed == "Vypočítat" else pd.to_numeric(df[col_speed].astype(str).str.replace(',', '.'), errors='coerce')
    
    df_valid = df[df['speed_kmh'] >= min_speed].copy()
    if not df_valid.empty:
        dir_cond = df_valid[col_dir] != df_valid[col_dir].shift().bfill() if col_dir in df_valid.columns else False
        time_gap = df_valid['parsed_time'].diff().dt.total_seconds() > 15
        df_valid['pass_id'] = (time_gap | dir_cond).cumsum() + 1
        return df_valid
    return pd.DataFrame()

# --- UI ---
with st.sidebar:
    st.header("📂 1. Data")
    uploaded_file = st.file_uploader("Nahrát CSV", type=['csv'])
    
    if uploaded_file:
        df_raw = nacti_surova_data(uploaded_file.getvalue())
        col_time = st.selectbox("Čas", df_raw.columns, index=df_raw.columns.get_loc(najdi_sloupec(df_raw.columns, ['time'])))
        col_lat = st.selectbox("Latitude", df_raw.columns, index=df_raw.columns.get_loc(najdi_sloupec(df_raw.columns, ['lat'])))
        col_lon = st.selectbox("Longitude", df_raw.columns, index=df_raw.columns.get_loc(najdi_sloupec(df_raw.columns, ['lon'])))
        col_stiff = st.selectbox("Tuhost (Kb)", df_raw.columns, index=df_raw.columns.get_loc(najdi_sloupec(df_raw.columns, ['stiff', 'kb', 'cmv'])))
        col_vib = st.selectbox("Amplituda/Vib", [None] + list(df_raw.columns), index=0)
        
        target_lat = st.number_input("Zkouška Y (Lat)", value=49.2793000, format="%.7f")
        target_lon = st.number_input("Zkouška X (Lon)", value=17.0212000, format="%.7f")
        
        st.header("📐 Rozměry stroje")
        offset_fwd = st.number_input("Podélný posun (m)", value=2.00)
        offset_right = st.number_input("Příčný posun (m)", value=0.20)
        roller_width = st.number_input("Šířka běhounu (m)", value=2.10)

if uploaded_file is not None:
    df = zpracuj_kinematiku(df_raw, col_lat, col_lon, col_time, "Vypočítat", None, offset_fwd, offset_right, 0.5, "1")
    
    # Převod senzorů
    df['Kb'] = pd.to_numeric(df_raw[col_stiff].astype(str).str.replace(',', '.'), errors='coerce')
    df['Amp'] = pd.to_numeric(df_raw[col_vib].astype(str).str.replace(',', '.'), errors='coerce') if col_vib else np.nan
    df['Bod_ID'] = df.index + 1
    
    geod = Geod(ellps="WGS84")
    
    # KROK 1: Které pojezdy trefily zkoušku?
    zasazene_pojezdy = []
    
    for p_id, group in df.groupby('pass_id'):
        _, _, dists = geod.inv(group['drum_lon'].values, group['drum_lat'].values, np.full(len(group), target_lon), np.full(len(group), target_lat))
        idx_cpa = np.argmin(dists)
        cpa_row = group.iloc[idx_cpa]
        radial_dist = dists[idx_cpa]
        
        # Výpočet příčné odchylky od osy (Cross-track)
        az_to_target, _, _ = geod.inv(cpa_row['drum_lon'], cpa_row['drum_lat'], target_lon, target_lat)
        angle_diff = math.radians(az_to_target - cpa_row['heading'])
        cross_track = abs(radial_dist * math.sin(angle_diff))
        
        # Spadá zkouška do běhounu? (Polovina šířky)
        if cross_track <= (roller_width / 2) and radial_dist < 5.0:
            zasazene_pojezdy.append((p_id, cross_track))

    if not zasazene_pojezdy:
        st.error(f"Zkoušku (v toleranci pásu {roller_width}m) netrefil žádný pojezd stroje.")
    else:
        # Vybereme ten nejvystředěnější pojezd (nebo necháme uživatele vybrat)
        best_pass_id = min(zasazene_pojezdy, key=lambda x: x[1])[0]
        st.success(f"Zkouška leží v ose pojezdu č. **{best_pass_id}** (Příčná odchylka osy zkoušky: {min(zasazene_pojezdy, key=lambda x: x[1])[1]:.2f} m).")
        
        df_pass = df[df['pass_id'] == best_pass_id].copy()
        
        # Oříznutí na body +- 5 metrů pro přehlednost UI
        _, _, local_dists = geod.inv(df_pass['drum_lon'].values, df_pass['drum_lat'].values, np.full(len(df_pass), target_lon), np.full(len(df_pass), target_lat))
        df_local = df_pass[local_dists <= 5.0].copy()
        df_local.insert(0, "Vybrat", False)

        # Tabulka s přesností na 7 míst
        df_display = df_local[['Vybrat', 'Bod_ID', 'parsed_time', 'drum_lon', 'drum_lat', 'Kb', 'Amp']].copy()
        df_display['drum_lon'] = df_display['drum_lon'].map('{:.7f}'.format)
        df_display['drum_lat'] = df_display['drum_lat'].map('{:.7f}'.format)
        
        col1, col2 = st.columns([1.2, 1])
        
        with col2:
            st.markdown("#### 🛠️ Výběr hran logu")
            st.caption("Vyber přesně 2 po sobě jdoucí body z tohoto pojezdu.")
            
            edited_df = st.data_editor(
                df_display, hide_index=True, use_container_width=True,
                disabled=['Bod_ID', 'parsed_time', 'drum_lon', 'drum_lat', 'Kb', 'Amp']
            )
            
            selected_rows = edited_df[edited_df['Vybrat'] == True]
            if len(selected_rows) == 2:
                final_kb = selected_rows['Kb'].mean()
                final_amp = selected_rows['Amp'].mean() if col_vib else "N/A"
                st.success(f"✅ **Aproximované Kb: {final_kb:.2f}**")
                if col_vib: st.info(f"Aproximovaná Amplituda: {final_amp:.2f}")
            elif len(selected_rows) > 0:
                st.warning(f"Vybráno bodů: {len(selected_rows)}. Pro aproximaci úsečky vyber přesně 2.")

        with col1:
            st.markdown("#### 🗺️ Stopa válce a body")
            fig = go.Figure()
            
            # Vykreslení HRAN BĚHOUNU (Pás)
            hl_lon, hl_lat = [], []
            hr_lon, hr_lat = [], []
            
            for _, row in df_local.iterrows():
                # Výpočet levého a pravého okraje pro každý bod
                l_lon, l_lat, _ = geod.fwd(row['drum_lon'], row['drum_lat'], (row['heading'] - 90) % 360, roller_width / 2)
                r_lon, r_lat, _ = geod.fwd(row['drum_lon'], row['drum_lat'], (row['heading'] + 90) % 360, roller_width / 2)
                hl_lon.append(l_lon); hl_lat.append(l_lat)
                hr_lon.append(r_lon); hr_lat.append(r_lat)
                
            # Spojení do polygonu pro vykreslení pruhu
            poly_lon = hl_lon + hr_lon[::-1] + [hl_lon[0]]
            poly_lat = hl_lat + hr_lat[::-1] + [hl_lat[0]]
            
            fig.add_trace(go.Scatter(
                x=poly_lon, y=poly_lat, mode='lines', fill='toself',
                fillcolor='rgba(0, 150, 136, 0.2)', line=dict(color='teal', width=1.5, dash='dash'),
                name=f'Pás běhounu ({roller_width}m)'
            ))
            
            # Osa pojezdu
            fig.add_trace(go.Scatter(
                x=df_local['drum_lon'], y=df_local['drum_lat'], mode='lines', 
                line=dict(color='teal', width=3), name='Osa Válce'
            ))
            
            # Samotné body logu (střed běhounu)
            fig.add_trace(go.Scatter(
                x=df_local['drum_lon'], y=df_local['drum_lat'], mode='markers+text',
                marker=dict(size=9, color='black'),
                text=df_local['Bod_ID'], textposition="bottom center", name='Záznamy (Bod_ID)',
                hovertext=[f"ID: {r['Bod_ID']}<br>Kb: {r['Kb']}" for _, r in df_local.iterrows()]
            ))
            
            # Zvýraznění vybraných dvou bodů
            if len(selected_rows) > 0:
                sel_ids = selected_rows['Bod_ID'].tolist()
                sel_pts = df_local[df_local['Bod_ID'].isin(sel_ids)]
                fig.add_trace(go.Scatter(
                    x=sel_pts['drum_lon'], y=sel_pts['drum_lat'], mode='markers+lines',
                    marker=dict(size=14, color='orange', symbol='circle-open', line=dict(width=3)),
                    line=dict(color='orange', width=2), name='Vybraná úsečka'
                ))
            
            # Bod Zkoušky
            fig.add_trace(go.Scatter(
                x=[target_lon], y=[target_lat], mode='markers',
                marker=dict(size=16, symbol='x', color='red', line=dict(width=3)),
                name='Bod Zkoušky'
            ))
            
            cos_corr = 1 / np.cos(np.radians(target_lat))
            fig.update_layout(
                yaxis=dict(scaleanchor="x", scaleratio=cos_corr, tickformat=".7f"),
                xaxis=dict(tickformat=".7f"), margin=dict(l=0, r=0, t=30, b=0), height=550,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)