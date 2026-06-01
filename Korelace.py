import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyproj import Geod
import io
import csv
import math

st.set_page_config(page_title="CCC Historie a Analýza Zkoušky v4", layout="wide")
st.title("📈 CCC: Komplexní historie pojezdů (Opravená Kinematika)")
st.caption("Globální přehled -> Lokální detail -> Chronologická historie hutnění na bodu zkoušky.")

# --- PARSOVÁNÍ DAT ---
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
    st.header("📂 1. Vstupní data ze stroje")
    uploaded_file = st.file_uploader("Nahrát CSV z válce", type=['csv'])
    
    if uploaded_file:
        df_raw = nacti_surova_data(uploaded_file.getvalue())
        col_time = st.selectbox("Čas", df_raw.columns, index=df_raw.columns.get_loc(najdi_sloupec(df_raw.columns, ['time', 'cas'])))
        col_lat = st.selectbox("Latitude", df_raw.columns, index=df_raw.columns.get_loc(najdi_sloupec(df_raw.columns, ['lat'])))
        col_lon = st.selectbox("Longitude", df_raw.columns, index=df_raw.columns.get_loc(najdi_sloupec(df_raw.columns, ['lon'])))
        col_stiff = st.selectbox("Tuhost (Kb)", df_raw.columns, index=df_raw.columns.get_loc(najdi_sloupec(df_raw.columns, ['stiff', 'kb', 'cmv'])))
        col_vib = st.selectbox("Amplituda/Vibrace", [None] + list(df_raw.columns), index=0)
        
        # NOVÉ: Parametry pro směr jízdy kvůli offsetům
        col_dir = st.selectbox("Směr jízdy (Důležité pro offset!)", [None] + list(df_raw.columns), index=list(df_raw.columns).index(najdi_sloupec(df_raw.columns, ['dir', 'smer'])) + 1 if najdi_sloupec(df_raw.columns, ['dir', 'smer']) else 0)
        forward_val = st.text_input("Znak/Číslo pro jízdu VPŘED (např. 1 nebo F)", value="1")
        
        st.header("🎯 2. Místo zkoušky")
        target_lat = st.number_input("Zkouška Y (Lat)", value=50.0791600, format="%.7f")
        target_lon = st.number_input("Zkouška X (Lon)", value=14.5930200, format="%.7f")
        
        st.header("📐 3. Stroj a senzory")
        offset_fwd = st.number_input("Podélný posun antény (m)", value=2.65, step=0.05)
        offset_right = st.number_input("Příčný posun antény (m)", value=0.26, step=0.05)
        roller_width = st.number_input("Šířka běhounu (m)", value=2.10, step=0.05)

# --- JÁDRO A VÝPOČTY ---
if uploaded_file is not None:
    df = df_raw.copy()
    df['lat'] = pd.to_numeric(df[col_lat].astype(str).str.replace(',', '.'), errors='coerce')
    df['lon'] = pd.to_numeric(df[col_lon].astype(str).str.replace(',', '.'), errors='coerce')
    df['Kb'] = pd.to_numeric(df[col_stiff].astype(str).str.replace(',', '.'), errors='coerce')
    df['Amp'] = pd.to_numeric(df[col_vib].astype(str).str.replace(',', '.'), errors='coerce') if col_vib else np.nan
    df['parsed_time'] = pd.to_datetime(df[col_time].astype(str).str.replace(' GMT', ''), utc=True, format='mixed', errors='coerce')
    df = df.dropna(subset=['lat', 'lon', 'parsed_time', 'Kb']).sort_values('parsed_time').reset_index(drop=True)
    df['Bod_ID'] = df.index + 1

    geod = Geod(ellps="WGS84")

    # 1. HLAVNÍ KINEMATIKA: Směr pohybu z GPS
    df['smooth_lon'] = df['lon'].rolling(3, min_periods=1, center=True).mean()
    df['smooth_lat'] = df['lat'].rolling(3, min_periods=1, center=True).mean()
    fwd_az, _, _ = geod.inv(df['smooth_lon'].shift().bfill().values, df['smooth_lat'].shift().bfill().values, df['smooth_lon'].values, df['smooth_lat'].values)
    gps_heading = fwd_az % 360
    
    # 2. DETEKCE POJEZDŮ (Podle pohybu GPS a času)
    df['dt'] = df['parsed_time'].diff().dt.total_seconds().fillna(0)
    diff_h = (gps_heading - pd.Series(gps_heading).shift().bfill()) % 360
    dir_change = (np.minimum(diff_h, 360 - diff_h) > 90) # Pokud se směr pohybu změní o > 90 stupňů, válec začal couvat
    df['pass_id'] = ((df['dt'] > 10) | dir_change).cumsum() + 1
    
    # 3. OPRAVA OFFSETŮ: Fyzické natočení stroje
    if col_dir and col_dir in df.columns:
        is_fwd = (df[col_dir].astype(str).str.strip() == str(forward_val)).values
        # Pokud couvá, čumák stroje zůstává natočený opačně než směr pohybu
        df['machine_heading'] = np.where(is_fwd, gps_heading, (gps_heading + 180) % 360)
    else:
        df['machine_heading'] = gps_heading # Nouzovka, pokud nemáš data o směru (bude to dělat ty chyby v offsetech)
        st.warning("Není vybrán sloupec 'Směr jízdy'. Příčný offset se bude při couvání překlápět na druhou stranu!")

    # 4. PROMÍTNUTÍ NA STŘED VÁLCE (Pomocí správného fyzického natočení stroje)
    df['mid_lon'], df['mid_lat'], _ = geod.fwd(df['lon'].values, df['lat'].values, df['machine_heading'].values, np.full(len(df), offset_fwd))
    h_right = (df['machine_heading'] - 90) % 360 if offset_right < 0 else (df['machine_heading'] + 90) % 360
    df['drum_lon'], df['drum_lat'], _ = geod.fwd(df['mid_lon'].values, df['mid_lat'].values, h_right, np.full(len(df), abs(offset_right)))

    # IDENTIFIKACE HISTORIE (Zásahy zkoušky)
    historie_pojezdu = []
    df_zasazene = pd.DataFrame() 

    for p_id, group in df.groupby('pass_id'):
        _, _, dists = geod.inv(group['drum_lon'].values, group['drum_lat'].values, np.full(len(group), target_lon), np.full(len(group), target_lat))
        idx_cpa = np.argmin(dists)
        radial_dist = dists[idx_cpa]
        
        # Pojistka: bod musí být aspoň do 5 metrů
        if radial_dist < 5.0:
            cpa_row = group.iloc[idx_cpa]
            az_to_target, _, _ = geod.inv(cpa_row['drum_lon'], cpa_row['drum_lat'], target_lon, target_lat)
            # Příčná odchylka se počítá od fyzického natočení stroje
            cross_track = abs(radial_dist * math.sin(math.radians(az_to_target - cpa_row['machine_heading'])))
            
            # Trefil válec zkoušku?
            if cross_track <= (roller_width / 2):
                # Najdeme 2 nejbližší body pro aproximaci Kb (z tohoto pojezdu)
                group_local = group.copy()
                group_local['dist_to_test_m'] = dists
                kandidati = group_local.nsmallest(2, 'dist_to_test_m')
                avg_kb = kandidati['Kb'].mean()
                
                historie_pojezdu.append({
                    'Real_Pass_ID': p_id,
                    'Čas protnutí': cpa_row['parsed_time'],
                    'Kolmá odchylka (m)': round(cross_track, 2),
                    'Aproximované Kb': round(avg_kb, 1),
                    'Amplituda': round(cpa_row['Amp'], 2) if col_vib else "N/A"
                })
                
                cpa_time = cpa_row['parsed_time']
                vyrez = group[
                    (group['parsed_time'] >= cpa_time - pd.Timedelta(seconds=5)) & 
                    (group['parsed_time'] <= cpa_time + pd.Timedelta(seconds=5))
                ].copy()
                vyrez['Is_Zasah'] = True
                vyrez['Real_Pass_ID'] = p_id
                df_zasazene = pd.concat([df_zasazene, vyrez])

    # --- UI LAYOUT ---
    if not historie_pojezdu:
        st.error("Žádný pojezd v datech netrefil zkoušku (osy byly příliš daleko).")
    else:
        df_hist = pd.DataFrame(historie_pojezdu).sort_values('Čas protnutí').reset_index(drop=True)
        df_hist.insert(0, 'Pořadí zkoušky', df_hist.index + 1)
        
        tab1, tab2, tab3 = st.tabs(["🌍 1. Globální Mapa", "🔍 2. Lokální Lupa", "📈 3. Historie a Vývoj Kb"])
        cos_corr = 1 / np.cos(np.radians(target_lat))
        color_seq = ['#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']

        with tab1:
            st.subheader("Globální pohled na stavbu")
            fig_global = go.Figure()
            
            fig_global.add_trace(go.Scattergl(
                x=df['drum_lon'], y=df['drum_lat'], mode='lines', 
                line=dict(color='lightgrey', width=1), name='Ostatní provoz'
            ))
            
            for idx, row in df_hist.iterrows():
                p_id = row['Real_Pass_ID']
                c = color_seq[idx % len(color_seq)]
                p_data = df_zasazene[df_zasazene['Real_Pass_ID'] == p_id]
                
                hl_lon, hl_lat, hr_lon, hr_lat = [], [], [], []
                for _, r in p_data.iterrows():
                    l_lon, l_lat, _ = geod.fwd(r['drum_lon'], r['drum_lat'], (r['machine_heading'] - 90) % 360, roller_width / 2)
                    r_lon, r_lat, _ = geod.fwd(r['drum_lon'], r['drum_lat'], (r['machine_heading'] + 90) % 360, roller_width / 2)
                    hl_lon.append(l_lon); hl_lat.append(l_lat)
                    hr_lon.append(r_lon); hr_lat.append(r_lat)
                
                poly_lon = hl_lon + hr_lon[::-1] + [hl_lon[0]]
                poly_lat = hl_lat + hr_lat[::-1] + [hl_lat[0]]
                
                fig_global.add_trace(go.Scatter(
                    x=poly_lon, y=poly_lat, mode='lines', fill='toself',
                    fillcolor=c.replace(')', ', 0.2)').replace('rgb', 'rgba') if 'rgb' in c else c, 
                    opacity=0.3, line=dict(color=c, width=1.5), name=f'{idx+1}. Hutnění (Pás)'
                ))
                
                fig_global.add_trace(go.Scatter(
                    x=p_data['drum_lon'], y=p_data['drum_lat'], mode='lines', 
                    line=dict(color=c, width=3), name=f'{idx+1}. Hutnění (Osa)'
                ))

            fig_global.add_trace(go.Scatter(
                x=[target_lon], y=[target_lat], mode='markers',
                marker=dict(size=14, symbol='x', color='black', line=dict(width=3)), name='Zkouška'
            ))

            fig_global.update_layout(yaxis=dict(scaleanchor="x", scaleratio=cos_corr), height=600, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_global, use_container_width=True)

        with tab2:
            st.subheader("Lokální detailní průjezdy")
            col_map, col_data = st.columns([1.5, 1])
            
            with col_map:
                fig_local = go.Figure()
                for idx, row in df_hist.iterrows():
                    p_id = row['Real_Pass_ID']
                    c = color_seq[idx % len(color_seq)]
                    p_data = df_zasazene[df_zasazene['Real_Pass_ID'] == p_id]
                    
                    fig_local.add_trace(go.Scatter(
                        x=p_data['drum_lon'], y=p_data['drum_lat'], mode='lines+markers+text',
                        marker=dict(size=8, color=c), line=dict(color=c, width=2),
                        text=p_data['Kb'], textposition="top center",
                        name=f'{idx+1}. Průjezd'
                    ))

                fig_local.add_trace(go.Scatter(
                    x=[target_lon], y=[target_lat], mode='markers',
                    marker=dict(size=16, symbol='x', color='black', line=dict(width=3)), name='Zkouška'
                ))

                buf = 0.0001
                fig_local.update_layout(
                    yaxis=dict(scaleanchor="x", scaleratio=cos_corr, tickformat=".7f", range=[target_lat-buf, target_lat+buf]),
                    xaxis=dict(tickformat=".7f", range=[target_lon-buf, target_lon+buf]),
                    height=550, margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_local, use_container_width=True)
                
            with col_data:
                st.markdown("#### Surová data zúčastněných bodů")
                df_table = df_zasazene[['Real_Pass_ID', 'Bod_ID', 'parsed_time', 'Kb', 'drum_lon', 'drum_lat']].copy()
                df_table['parsed_time'] = df_table['parsed_time'].dt.strftime('%H:%M:%S')
                df_table['drum_lon'] = df_table['drum_lon'].map('{:.7f}'.format)
                df_table['drum_lat'] = df_table['drum_lat'].map('{:.7f}'.format)
                st.dataframe(df_table, hide_index=True, use_container_width=True, height=500)

        with tab3:
            st.subheader("📊 Teoretické pořadí hodnot a Vývoj Tuhosti")
            col_chart, col_hist = st.columns([1.5, 1])
            
            with col_hist:
                df_hist['Čas protnutí'] = df_hist['Čas protnutí'].dt.strftime('%H:%M:%S')
                st.dataframe(df_hist, hide_index=True, use_container_width=True)
                
                st.metric("Finální Tuhost z posledního pojezdu", f"{df_hist.iloc[-1]['Aproximované Kb']} [-]")

            with col_chart:
                fig_curve = go.Figure()
                fig_curve.add_trace(go.Scatter(
                    x=df_hist['Pořadí zkoušky'], y=df_hist['Aproximované Kb'],
                    mode='markers+lines',
                    marker=dict(size=14, color='teal', symbol='diamond', line=dict(width=2, color='black')),
                    line=dict(width=3, color='teal', dash='solid'),
                    name="Osová tuhost stopy"
                ))
                fig_curve.update_layout(
                    title="Nárůst tuhosti v místě zkoušky (Zprůměrovány 2 nejbližší body pojezdu)",
                    xaxis_title="Pořadí přejetí",
                    yaxis_title="Hodnota Kb [-]",
                    xaxis=dict(dtick=1)
                )
                st.plotly_chart(fig_curve, use_container_width=True)
else:
    st.info("Nahraj data. A nezapomeň nastavit sloupec pro 'Směr jízdy'!")
