# ----------------------------------------------------------
# app.py — Versión MEJORADA (OCR robusto universal + Parsers inteligentes)
# ----------------------------------------------------------

import os
import io
import re
import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from fitparse import FitFile
from datetime import datetime, timedelta, time
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# Configuración Streamlit
# -------------------------------------------------------------------
st.set_page_config(page_title="Análisis Profesional Ciclismo", layout="wide")
st.title("🚴‍♂️ Análisis Profesional de Rendimiento Ciclista — OCR Inteligente 2025")

# (Opcional) Si necesitas especificar ruta de Tesseract en Windows, descomenta:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------------------------------------------------------
# Parámetros
# -------------------------------------------------------------------
st.sidebar.header("Parámetros de análisis")
weight = st.sidebar.number_input("Peso ciclista (kg)", min_value=30.0, max_value=120.0, value=70.0, step=0.5)
ftp_input = st.sidebar.number_input("FTP (W)", min_value=0.0, max_value=2000.0, value=250.0, step=1.0)

st.sidebar.header("🔧 Parámetros de Segmentación (altimetría)")
min_segment_seconds = st.sidebar.number_input("Duración mínima segmento (s)", min_value=30, max_value=600, value=120, step=10)
grade_threshold = st.sidebar.slider("Umbral de pendiente (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
min_elevation_gain = st.sidebar.number_input("Desnivel mínimo por segmento (m)", min_value=5, max_value=50, value=15, step=5)
smoothing_window = st.sidebar.number_input("Ventana suavizado altitud (puntos)", min_value=5, max_value=51, value=15, step=2)

# -------------------------------------------------------------------
# Utilidades FIT y análisis
# -------------------------------------------------------------------
def safe_col_contains(df, candidates):
    for c in df.columns:
        lc = str(c).lower()
        for cand in candidates:
            if cand.lower() in lc:
                return c
    return None

def read_fit(file):
    fitfile = FitFile(file)
    rows = []
    for msg in fitfile.get_messages("record"):
        row = {}
        for d in msg:
            row[str(d.name).lower()] = d.value
        rows.append(row)
    df = pd.DataFrame(rows)
    df.columns = [str(c).lower() for c in df.columns]
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def compute_np(series_power, timestamps):
    try:
        p = pd.Series(series_power).fillna(0).astype(float)
        if len(timestamps) < 2:
            return float(p.mean())
        dt = np.median(np.diff(np.array(timestamps).astype('int64') // 1_000_000_000))
        window = max(1, int(round(30.0 / dt))) if dt > 0 else 1
        mov = p.rolling(window=window, min_periods=1).mean()
        np_val = (np.mean(mov**4)) ** 0.25
        return float(np_val)
    except Exception:
        return float(series_power.mean())

def smooth_altitude(altitude_series, window_size=15):
    if len(altitude_series) < window_size:
        return altitude_series
    try:
        w = min(window_size, len(altitude_series) - 1)
        if w % 2 == 0:
            w -= 1
        return savgol_filter(altitude_series, window_length=w, polyorder=2)
    except Exception:
        return pd.Series(altitude_series).rolling(window=window_size, center=True, min_periods=1).mean().values

def compute_grade_percent(df):
    if 'altitude' not in df.columns:
        return pd.Series(0.0, index=df.index)
    d2 = df.copy()
    d2['alt_diff'] = d2['altitude'].diff().fillna(0)
    dist_col = safe_col_contains(d2, ['distance','distance_'])
    if dist_col and dist_col in d2.columns:
        d2['dist_diff'] = d2[dist_col].diff().replace(0, np.nan)
        grade = (d2['alt_diff'] / d2['dist_diff']).fillna(0) * 100
    else:
        if 'speed' in d2.columns and 'timestamp' in d2.columns:
            d2['dt'] = d2['timestamp'].diff().dt.total_seconds().replace(0, np.nan)
            d2['dist_est'] = d2['speed'] * d2['dt']
            grade = (d2['alt_diff'] / d2['dist_est']).fillna(0) * 100
        else:
            grade = pd.Series(0.0, index=d2.index)
    return grade.replace([np.inf, -np.inf], 0).fillna(0)

def compute_grade_from_smooth_altitude(df):
    if 'altitude' not in df.columns or 'timestamp' not in df.columns:
        return pd.Series(0.0, index=df.index)
    d = df.copy()
    d['altitude_smooth'] = smooth_altitude(d['altitude'].values, smoothing_window)
    d['alt_diff'] = pd.Series(d['altitude_smooth']).diff().fillna(0)
    if 'speed' in d.columns:
        d['time_diff'] = d['timestamp'].diff().dt.total_seconds().fillna(1)
        d['dist_est'] = d['speed'] * d['time_diff']
        g = (d['alt_diff'] / d['dist_est']).replace([np.inf, -np.inf], 0).fillna(0) * 100
    else:
        g = pd.Series(0.0, index=d.index)
    return g

def detect_climbs_from_elevation(df, min_gain=15, min_duration=60, thr=2.0):
    if 'altitude' not in df.columns or 'timestamp' not in df.columns:
        return pd.Series(['Llano'] * len(df), index=df.index)
    d = df.copy()
    d['altitude_smooth'] = smooth_altitude(d['altitude'].values, smoothing_window)
    d['grade_smooth'] = compute_grade_from_smooth_altitude(d)
    climb_mask = (d['grade_smooth'] > thr)
    d['climb_start'] = (climb_mask & ~climb_mask.shift(1).fillna(False))
    labels = ['Llano'] * len(d)
    current = None
    for i in range(len(d)):
        if current is None and d.loc[i, 'climb_start']:
            current = {'start': i, 'gain': 0}
        if current is not None:
            gain = d.loc[i, 'altitude_smooth'] - d.loc[current['start'], 'altitude_smooth']
            duration = (d.loc[i, 'timestamp'] - d.loc[current['start'], 'timestamp']).total_seconds()
            end_cond = (d.loc[i, 'grade_smooth'] < thr * 0.5) and (gain >= min_gain) and (duration >= min_duration)
            last = (i == len(d) - 1)
            if end_cond or last:
                if gain >= min_gain and duration >= min_duration:
                    labels[current['start']:i+1] = ['Subida'] * (i - current['start'] + 1)
                current = None
    return pd.Series(labels, index=df.index)

def merge_short_segments(df, seg_col='segment', min_seconds=60):
    if 'timestamp' not in df.columns:
        return df
    d = df.copy().reset_index(drop=True)
    d['seg_id'] = (d[seg_col] != d[seg_col].shift(1)).cumsum()
    durations = d.groupby('seg_id').apply(lambda x: (x['timestamp'].iloc[-1] - x['timestamp'].iloc[0]).total_seconds())
    short_ids = durations[durations < min_seconds].index.tolist()
    for sid in short_ids:
        ids = sorted(d['seg_id'].unique())
        idx = ids.index(sid)
        cur_type = d[d['seg_id'] == sid][seg_col].iloc[0]
        left_type = d[d['seg_id'] == ids[idx-1]][seg_col].iloc[0] if idx > 0 else None
        right_type = d[d['seg_id'] == ids[idx+1]][seg_col].iloc[0] if idx < len(ids)-1 else None
        if left_type == cur_type:
            d.loc[d['seg_id'] == sid, 'seg_id'] = ids[idx-1]
        elif right_type == cur_type:
            d.loc[d['seg_id'] == sid, 'seg_id'] = ids[idx+1]
        else:
            if idx == 0 and len(ids) > 1:
                d.loc[d['seg_id'] == sid, 'seg_id'] = ids[1]
            elif idx == len(ids)-1 and len(ids) > 1:
                d.loc[d['seg_id'] == sid, 'seg_id'] = ids[-2]
            elif len(ids) > 2:
                left_d = durations.loc[ids[idx-1]]
                right_d = durations.loc[ids[idx+1]]
                d.loc[d['seg_id'] == sid, 'seg_id'] = ids[idx-1] if left_d >= right_d else ids[idx+1]
    seg_map = d.groupby('seg_id')[seg_col].agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]).to_dict()
    d[seg_col] = d['seg_id'].map(seg_map)
    d.drop(columns=['seg_id'], inplace=True)
    return d

def smart_segmentation(df, min_segment_seconds=120, min_elevation_gain=15, grade_threshold=2.0):
    base = detect_climbs_from_elevation(df, min_elevation_gain, min_segment_seconds, grade_threshold)
    d = df.copy()
    d['grade_pct'] = compute_grade_percent(d)
    remaining = (base == 'Llano')
    labels_rem = np.where(d.loc[remaining, 'grade_pct'] < -grade_threshold, 'Bajada', 'Llano')
    final = base.copy()
    final[remaining] = labels_rem
    d['segment'] = final
    d = merge_short_segments(d, 'segment', min_seconds=min_segment_seconds)
    return d['segment']

def summarize_segments(df, weight_kg):
    if 'segment' not in df.columns or 'timestamp' not in df.columns or len(df) == 0:
        return pd.DataFrame()
    d = df.sort_values('timestamp').reset_index(drop=True)
    d['seg_block'] = (d['segment'] != d['segment'].shift(1)).cumsum()
    rows = []
    for sid, g in d.groupby('seg_block'):
        seg_type = g['segment'].iloc[0]
        t0, t1 = g['timestamp'].iloc[0], g['timestamp'].iloc[-1]
        duration_min = (t1 - t0).total_seconds() / 60.0
        if duration_min < 0.5:
            continue
        dist_m = 0.0
        dist_col = safe_col_contains(g, ['distance','distance_'])
        if dist_col and dist_col in g.columns:
            try:
                dist_m = float(g[dist_col].iloc[-1] - g[dist_col].iloc[0])
            except:
                pass
        elevation_gain = 0.0
        if 'altitude' in g.columns and seg_type == 'Subida':
            elevation_gain = float(g['altitude'].iloc[-1] - g['altitude'].iloc[0])
        avg_power = float(g['power'].dropna().mean()) if 'power' in g.columns and not g['power'].dropna().empty else np.nan
        max_power = float(g['power'].dropna().max()) if 'power' in g.columns and not g['power'].dropna().empty else np.nan
        np_val = compute_np(g['power'].fillna(0), g['timestamp']) if 'power' in g.columns else np.nan
        avg_hr = float(g['heart_rate'].dropna().mean()) if 'heart_rate' in g.columns and not g['heart_rate'].dropna().empty else np.nan
        avg_speed_kmh = float(g['speed'].dropna().mean()*3.6) if 'speed' in g.columns and not g['speed'].dropna().empty else np.nan
        avg_wkg = avg_power / weight_kg if not np.isnan(avg_power) else np.nan
        np_wkg = np_val / weight_kg if not np.isnan(np_val) else np.nan
        rows.append({
            'segment_id': int(sid),
            'tipo': seg_type,
            'start_time': t0,
            'end_time': t1,
            'duration_min': round(duration_min, 2),
            'dist_m': round(dist_m, 1),
            'elevation_gain_m': round(elevation_gain, 1) if seg_type == 'Subida' else 0,
            'power_avg_w': round(avg_power,1) if not np.isnan(avg_power) else np.nan,
            'power_max_w': round(max_power,1) if not np.isnan(max_power) else np.nan,
            'power_np_w': round(np_val,1) if not np.isnan(np_val) else np.nan,
            'np_wkg': round(np_wkg,2) if not np.isnan(np_wkg) else np.nan,
            'avg_wkg': round(avg_wkg,2) if not np.isnan(avg_wkg) else np.nan,
            'speed_avg_kmh': round(avg_speed_kmh,2) if not np.isnan(avg_speed_kmh) else np.nan,
            'hr_avg': round(avg_hr,1) if not np.isnan(avg_hr) else np.nan
        })
    return pd.DataFrame(rows)

# -------------------------------------------------------
#  OCR MEJORADO — PARSERS INTELIGENTES
# -------------------------------------------------------

def normalize_image(img: Image.Image, target_h=2000):
    """Reescala manteniendo proporción para mejor OCR."""
    w, h = img.size
    scale = target_h / h
    new_w = int(w * scale)
    img = img.resize((new_w, target_h), Image.LANCZOS)
    return img

def preprocess_for_ocr(bgr):
    """Preprocesamiento mejorado para OCR"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # CLAHE para contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Desenfoque suave + unsharp masking
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    
    # Umbral adaptativo más agresivo
    thr = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 41, 15)
    
    # Operaciones morfológicas para limpiar ruido
    kernel = np.ones((2, 2), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    
    return thr

def detect_keyword_y(full_bin, keywords):
    """Devuelve coordenada Y de cualquiera de las palabras clave."""
    df = pytesseract.image_to_data(full_bin, lang="eng+spa", 
                                   config="--oem 3 --psm 6",
                                   output_type=pytesseract.Output.DATAFRAME)
    df = df.dropna(subset=["text"])
    for keyword in keywords:
        m = df[df["text"].str.strip().str.upper() == keyword.upper()]
        if len(m):
            return int(m["top"].median())
    return None

def cut_blocks_by_keywords(bgr):
    """Corta HEADER / SPORTS / SPLITS por detección de palabras clave mejorada."""
    full_bin = preprocess_for_ocr(bgr)
    H, W = full_bin.shape

    # Buscar múltiples palabras clave para cada sección
    y_sports = detect_keyword_y(full_bin, ["SPORTS", "SPORT", "DEPORTES"])
    y_splits = detect_keyword_y(full_bin, ["SPLITS", "SPLIT", "TRAMOS", "PARCiales"])

    # Fallbacks inteligentes
    if y_sports is None:
        y_sports = int(H * 0.35)  # Más arriba para capturar mejor el header

    if y_splits is None or y_splits < y_sports:
        y_splits = int(H * 0.65)  # Más espacio para sports

    header = bgr[0: y_sports-10, :] if y_sports else bgr[0: int(H*0.35), :]
    sports = bgr[y_sports-10: y_splits, :] if y_sports and y_splits else bgr[int(H*0.35): int(H*0.65), :]
    splits = bgr[y_splits: H, :] if y_splits else bgr[int(H*0.65): H, :]

    return header, sports, splits

def ocr_text(img_bin):
    return pytesseract.image_to_string(img_bin, lang="eng+spa", config="--oem 3 --psm 6")

def ocr_data(img_bin):
    return pytesseract.image_to_data(img_bin, lang="eng+spa", config="--oem 3 --psm 6",
                                     output_type=pytesseract.Output.DATAFRAME)

# ------------------------
# PARSER DEL HEADER MEJORADO
# ------------------------

rx_time_hms = re.compile(r"\b(\d{1,2}:\d{2}:\d{2})\b")
rx_time_ms = re.compile(r"\b(\d{1,2}:\d{2})\b")
rx_place = re.compile(r"\b(\d{1,4})\s*°")
rx_speed = re.compile(r"(\d{1,2}[.,]\d{1,2})\s*km/h", re.IGNORECASE)
rx_hora = re.compile(r"\b(\d{1,2}:\d{2})\b")  # Solo hora:minutos para hora de inicio

def parse_header_mejorado(header_txt):
    """
    Parser inteligente que distingue entre:
    - Tiempo total (duración)
    - Hora de inicio (hora del día)
    - Posición
    - Ritmo
    """
    data = {
        "posicion": None,
        "tiempo_total": None,
        "ritmo_kmh": None,
        "ritmo_min_km": None,
        "tiempo_disparo": None,
        "hora_inicio_real": None,
        "categoria": None
    }
    
    lines = [l.strip() for l in header_txt.splitlines() if l.strip()]
    
    # Buscar posición (número seguido de °)
    for line in lines:
        m = rx_place.search(line)
        if m:
            data["posicion"] = m.group(1)
            break
    
    # Estrategia para tiempos:
    # 1. Buscar patrones específicos con etiquetas
    # 2. Distinguir entre duraciones y horas del día
    
    tiempos_hms = rx_time_hms.findall(header_txt)
    tiempos_ms = rx_time_ms.findall(header_txt)
    
    # Buscar hora de inicio (debe ser patrón HH:MM y estar cerca de palabras clave)
    hora_keywords = ["salida", "start", "inicio", "real", "hora"]
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(kw in line_lower for kw in hora_keywords):
            # Buscar HH:MM en esta línea o la siguiente
            m = rx_hora.search(line)
            if m:
                data["hora_inicio_real"] = m.group(1) + ":00"  # Agregar segundos
                break
            # Si no está en esta línea, buscar en las siguientes
            for next_line in lines[i+1:i+3]:
                m = rx_hora.search(next_line)
                if m:
                    data["hora_inicio_real"] = m.group(1) + ":00"
                    break
    
    # Buscar tiempo total (duracciones largas, típicamente > 1 hora)
    tiempo_keywords = ["tiempo", "total", "final", "dura", "time"]
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(kw in line_lower for kw in tiempo_keywords):
            # Buscar HH:MM:SS en esta línea o adyacentes
            for check_line in lines[max(0,i-1):i+2]:
                m = rx_time_hms.search(check_line)
                if m:
                    # Verificar si es una duración razonable (más de 1 minuto)
                    h, m, s = map(int, m.group(1).split(':'))
                    total_sec = h*3600 + m*60 + s
                    if total_sec > 60:  # Más de 1 minuto
                        data["tiempo_total"] = m.group(1)
                        break
    
    # Si no se encontró por keywords, usar heurística:
    # El primer HH:MM:SS largo es probablemente el tiempo total
    if not data["tiempo_total"] and tiempos_hms:
        for tiempo in tiempos_hms:
            h, m, s = map(int, tiempo.split(':'))
            if h > 0 or m >= 10:  # Duración significativa
                data["tiempo_total"] = tiempo
                break
    
    # Ritmo
    m = rx_speed.search(header_txt)
    if m:
        data["ritmo_kmh"] = m.group(1).replace(',', '.')
    
    # Categoría
    for line in lines:
        if any(kw in line.lower() for kw in ["categor", "cat.", "cat "]):
            data["categoria"] = line
            break
    
    return data

# ------------------------
# PARSER SPLITS MEJORADO
# ------------------------

def parse_splits_mejorado(df_words):
    """
    Parser inteligente para tabla de splits que:
    - Filtra ruido del OCR
    - Agrupa correctamente por líneas
    - Identifica etiquetas, tiempos y posiciones
    """
    if df_words is None or df_words.empty:
        return []
    
    df = df_words.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    
    if df.empty:
        return []
    
    df["right"] = df["left"] + df["width"]
    df["bottom"] = df["top"] + df["height"]
    
    # Agrupar por líneas (con tolerancia vertical)
    line_groups = []
    used_indices = set()
    
    for _, row in df.iterrows():
        if row.name in used_indices:
            continue
            
        # Encontrar todos los elementos en la misma línea
        same_line = df[
            (abs(df["top"] - row["top"]) < row["height"] * 0.8) &
            (~df.index.isin(used_indices))
        ].sort_values("left")
        
        if len(same_line) > 0:
            line_text = " ".join(same_line["text"].tolist())
            line_groups.append({
                "top": row["top"],
                "text": line_text,
                "elements": same_line
            })
            used_indices.update(same_line.index)
    
    # Ordenar líneas por posición vertical
    line_groups.sort(key=lambda x: x["top"])
    
    rows = []
    for line in line_groups:
        text = line["text"]
        
        # Filtrar líneas de ruido (demasiado cortas o sin tiempos)
        if len(text) < 3:
            continue
            
        # Buscar tiempos en la línea
        times_hms = rx_time_hms.findall(text)
        times_ms = rx_time_ms.findall(text)
        
        # Solo procesar líneas que contengan tiempos
        if not times_hms and not times_ms:
            continue
        
        # Identificar etiqueta (primer elemento no numérico/temporal)
        elements = line["elements"]
        etiqueta = None
        
        for elem in elements.itertuples():
            elem_text = elem.text.strip()
            # Es etiqueta si no es tiempo, no es posición y no es número
            if (not rx_time_hms.match(elem_text) and 
                not rx_time_ms.match(elem_text) and 
                not rx_place.match(elem_text) and
                not elem_text.replace('.', '').replace(',', '').isdigit() and
                "km/h" not in elem_text.lower()):
                etiqueta = elem_text
                break
        
        # Determinar tiempos
        tiempo_carrera = None
        tiempo_parcial = None
        
        if times_hms:
            if len(times_hms) >= 2:
                tiempo_carrera, tiempo_parcial = times_hms[0], times_hms[1]
            elif len(times_hms) == 1:
                tiempo_carrera = times_hms[0]
        elif times_ms:
            if len(times_ms) >= 2:
                tiempo_carrera, tiempo_parcial = f"00:{times_ms[0]}", f"00:{times_ms[1]}"
            elif len(times_ms) == 1:
                tiempo_carrera = f"00:{times_ms[0]}"
        
        # Buscar posición
        place_match = rx_place.search(text)
        place = place_match.group(1) if place_match else None
        
        # Validar que sea un split razonable
        if tiempo_carrera:
            # Verificar que el tiempo sea plausible (menos de 24 horas)
            try:
                if ':' in tiempo_carrera:
                    parts = tiempo_carrera.split(':')
                    if len(parts) == 3:
                        h, m, s = map(int, parts)
                        if h < 24 and m < 60 and s < 60:
                            rows.append({
                                "etiqueta": etiqueta,
                                "tiempo_carrera": tiempo_carrera,
                                "tiempo_parcial": tiempo_parcial,
                                "place": place
                            })
            except:
                continue
    
    # Filtrar duplicados y limpiar
    unique_rows = []
    seen = set()
    
    for row in rows:
        key = (row.get("etiqueta"), row.get("tiempo_carrera"))
        if key not in seen and key != (None, None):
            unique_rows.append(row)
            seen.add(key)
    
    return unique_rows

# ------------------------
# OCR COMPLETO MEJORADO
# ------------------------

def extract_ocr_mejorado(full_img):
    """
    Extracción OCR mejorada con parsers inteligentes
    """
    # Normalizar imagen
    img_norm = normalize_image(full_img, target_h=2200)
    bgr = cv2.cvtColor(np.array(img_norm), cv2.COLOR_RGB2BGR)
    
    # Segmentar por palabras clave
    header_bgr, sports_bgr, splits_bgr = cut_blocks_by_keywords(bgr)
    
    # Preprocesar cada región
    hbin = preprocess_for_ocr(header_bgr)
    sbin = preprocess_for_ocr(sports_bgr) 
    tbin = preprocess_for_ocr(splits_bgr)
    
    # OCR de cada región
    header_txt = ocr_text(hbin)
    sports_txt = ocr_text(sbin)
    splits_df = ocr_data(tbin)
    
    # Parsear con algoritmos mejorados
    header_data = parse_header_mejorado(header_txt)
    splits_rows = parse_splits_mejorado(splits_df)
    
    # Parseo simple para sports (mantenido igual)
    sports_rows = []
    lines = [l.strip() for l in sports_txt.splitlines() if l.strip()]
    cur = {}
    for line in lines:
        if any(x in line for x in ["Pioj", "La Negra", "Negra", "Piojó"]):
            if cur:
                sports_rows.append(cur)
            cur = {"nombre": line}
            continue
        m = rx_time_hms.search(line) or rx_time_ms.search(line)
        if m:
            cur["tiempo"] = m.group(1)
        m = rx_place.search(line)
        if m:
            cur["place"] = m.group(1)
        m = rx_speed.search(line)
        if m:
            cur["vel_kmh"] = m.group(1).replace(',', '.')
    if cur:
        sports_rows.append(cur)
    
    # Resultado final
    resultado = {
        "header_text": header_txt,
        "sports_text": sports_txt,
        "splits_rows": splits_rows,
        "sports_rows": sports_rows,
        "posicion": header_data.get("posicion"),
        "tiempo_total": header_data.get("tiempo_total"),
        "ritmo_promedio": header_data.get("ritmo_kmh"),
        "hora_inicio_real": header_data.get("hora_inicio_real"),
        "categoria": header_data.get("categoria"),
        "splits": [r["tiempo_carrera"] for r in splits_rows if r.get("tiempo_carrera")]
    }
    
    return resultado, (hbin, sbin, tbin)

# -------------------------------------------------------------------
# Ajuste de tiempos por OCR
# -------------------------------------------------------------------
def ajustar_tiempos_con_ocr(df_fit, datos_ocr):
    if not datos_ocr.get('hora_inicio_real'):
        return df_fit, "No se pudo detectar hora de inicio válida en la imagen"
    try:
        hms = datos_ocr['hora_inicio_real'].strip()
        # Asegurar formato HH:MM:SS
        if hms.count(':') == 1:
            hms = hms + ':00'
        h, m, s = map(int, hms.split(':'))
        if 'timestamp' in df_fit.columns and len(df_fit) > 0:
            fecha_fit = df_fit['timestamp'].iloc[0].date()
            hora_inicio_real = datetime.combine(fecha_fit, time(h, m, s))
            hora_inicio_fit = df_fit['timestamp'].iloc[0]
            diff = hora_inicio_real - hora_inicio_fit
            dfa = df_fit.copy()
            dfa['timestamp'] = dfa['timestamp'] + diff
            return dfa, f"✅ Tiempos ajustados. Hora real: {hora_inicio_real.strftime('%H:%M:%S')}"
        else:
            return df_fit, "No se pudo obtener fecha del archivo FIT"
    except Exception as e:
        return df_fit, f"Error ajustando tiempos: {str(e)}"

# -------------------------------------------------------------------
# Análisis de texto integrado
# -------------------------------------------------------------------
def generar_analisis_integrado_ocr(seg_summary, df_fit, ftp, weight, datos_ocr):
    A = []
    A.append("## 🏁 INFORMACIÓN OFICIAL DE LA CARRERA")
    if datos_ocr.get('hora_inicio_real'):
        A.append(f"**Hora de inicio oficial**: {datos_ocr['hora_inicio_real']}")
    if datos_ocr.get('tiempo_total'):
        A.append(f"**Tiempo oficial**: {datos_ocr['tiempo_total']}")
    if datos_ocr.get('posicion'):
        A.append(f"**Posición**: {datos_ocr['posicion']}°")
    if datos_ocr.get('ritmo_promedio'):
        A.append(f"**Ritmo promedio**: {datos_ocr['ritmo_promedio']} km/h")
    if datos_ocr.get('splits'):
        A.append(f"**Splits detectados**: {len(datos_ocr['splits'])}")

    A.append("\n## 📊 COMPARATIVA: OFICIAL vs CICLOCOMPUTADOR")
    if 'timestamp' in df_fit.columns and len(df_fit) > 0:
        t0, t1 = df_fit['timestamp'].iloc[0], df_fit['timestamp'].iloc[-1]
        dur_fit_min = (t1 - t0).total_seconds()/60
        A.append(f"**Duración según ciclocomputador**: {dur_fit_min:.1f} min")
        if datos_ocr.get('tiempo_total'):
            h, m, s = map(int, datos_ocr['tiempo_total'].split(':'))
            min_ocr = h*60 + m + s/60
            diff = abs(dur_fit_min - min_ocr)
            A.append(f"**Diferencia de tiempo**: {diff:.1f} min")
            if diff < 2:
                A.append("✅ **PRECISIÓN TEMPORAL**: Excelente")
            elif diff < 5:
                A.append("⚠️ **PRECISIÓN TEMPORAL**: Buena")
            else:
                A.append("🔍 **PRECISIÓN TEMPORAL**: Revisar sincronización")
    # Splits (listar primeros)
    if datos_ocr.get('splits'):
        A.append("\n## ⏱️ ANÁLISIS DE SPLITS OFICIALES")
        for i, sp in enumerate(datos_ocr['splits'][:6]):
            A.append(f"- Split {i+1}: {sp}")
    return "\n".join(A)

def generar_analisis_segmentos_contextual(seg_summary, df_fit, ftp, weight):
    if seg_summary.empty:
        return "No hay suficientes segmentos para análisis detallado."
    A = []
    A.append("## 🏆 SEGMENTOS CLAVE DE LA CARRERA")
    top_int = seg_summary.nlargest(3, 'np_wkg')
    if not top_int.empty:
        A.append("### 💥 Momentos de Mayor Intensidad")
        for _, s in top_int.iterrows():
            A.append(f"**{s['tipo']}** — {s['duration_min']:.1f} min")
            A.append(f"  - NP: {s['power_np_w']:.0f}W ({s['np_wkg']:.2f} W/kg)")
            if not pd.isna(s['speed_avg_kmh']):
                A.append(f"  - Vel.: {s['speed_avg_kmh']:.1f} km/h")
            if not pd.isna(s['hr_avg']):
                A.append(f"  - FC: {s['hr_avg']:.0f} bpm")
    longos = seg_summary.nlargest(2, 'duration_min')
    if not longos.empty:
        A.append("\n### ⏱️ Segmentos Más Largos")
        for _, s in longos.iterrows():
            dist_km = s['dist_m']/1000 if not pd.isna(s['dist_m']) else 0
            A.append(f"**{s['tipo']}** — {s['duration_min']:.1f} min | {dist_km:.2f} km | {s['power_avg_w']:.0f}W")

    A.append("\n## 🏔️ RENDIMIENTO POR TIPO DE TERRENO")
    for terr in ['Subida','Llano','Bajada']:
        tdf = seg_summary[seg_summary['tipo']==terr]
        if not tdf.empty:
            A.append(f"**{terr}**")
            A.append(f"  - Tiempo total: {tdf['duration_min'].sum():.1f} min")
            if not tdf['np_wkg'].dropna().empty:
                A.append(f"  - W/kg (NP) prom.: {tdf['np_wkg'].mean():.2f}")
            if not tdf['power_avg_w'].dropna().empty:
                A.append(f"  - Potencia prom.: {tdf['power_avg_w'].mean():.0f} W")

    if len(seg_summary) > 4:
        terc = len(seg_summary)//3
        ini = seg_summary.head(terc)['np_wkg'].mean()
        med = seg_summary.iloc[terc:terc*2]['np_wkg'].mean()
        fin = seg_summary.tail(terc)['np_wkg'].mean()
        A.append("\n## 🎯 ESTRATEGIA Y DISTRIBUCIÓN DEL ESFUERZO")
        A.append(f"- Inicio: {ini:.2f} W/kg | Medio: {med:.2f} W/kg | Final: {fin:.2f} W/kg")
        if fin > ini*1.1:
            A.append("🚀 Negative Split — excelente gestión")
        elif fin >= ini*0.9:
            A.append("⚖️ Pacing consistente")
    return "\n".join(A)

# -------------------------------------------------------------------
# Interfaz: carga de archivos
# -------------------------------------------------------------------
st.header("📁 Carga tus archivos")
st.info("""
**Instrucciones:**
1. **Archivo .FIT** (OBLIGATORIO): Sube el archivo de tu ciclocomputador (Garmin, Wahoo, etc.)
2. **Imagen de resultados** (opcional pero recomendado): Foto del ranking / resultados de la carrera
3. **TrainingPeaks** (opcional): Archivo CSV/Excel de TrainingPeaks para comparativa
""")

c1, c2, c3 = st.columns(3)
with c1:
    uploaded_image = st.file_uploader("Imagen resultados", type=["png","jpg","jpeg"])
with c2:
    fit_file = st.file_uploader("Archivo .FIT *", type=["fit"])
with c3:
    tp_file = st.file_uploader("TrainingPeaks", type=["csv","xlsx"])

# -------------------------------------------------------------------
# OCR MEJORADO
# -------------------------------------------------------------------
datos_ocr = {}
if uploaded_image:
    try:
        st.markdown("## 📷 Imagen de Resultados")
        im = Image.open(uploaded_image).convert('RGB')
        show = im.copy()
        show.thumbnail((700, 700))
        st.image(show, caption="Resultados oficiales", use_container_width=False)

        with st.spinner("Procesando imagen (OCR Inteligente + Parsers Mejorados)…"):
            datos_ocr, (hbin, sbin, tbin) = extract_ocr_mejorado(im)

        colA, colB, colC = st.columns(3)
        with colA:
            st.image(hbin, caption="Header binarizado", use_container_width=True, clamp=True)
        with colB:
            st.image(sbin, caption="Sports binarizado", use_container_width=True, clamp=True)
        with colC:
            st.image(tbin, caption="Splits binarizado", use_container_width=True, clamp=True)

        st.markdown("### 🔍 Información Extraída")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Posición", f"{datos_ocr.get('posicion','—')}°")
        with m2:
            st.metric("Tiempo", datos_ocr.get('tiempo_total', '—'))
        with m3:
            st.metric("Ritmo (km/h)", datos_ocr.get('ritmo_promedio', '—'))
        with m4:
            st.metric("Salida real", datos_ocr.get('hora_inicio_real', '—'))

        if datos_ocr.get('splits'):
            st.markdown("#### ⏱️ Splits (tiempo de carrera)")
            cols = st.columns(min(6, len(datos_ocr['splits'])))
            for i, sp in enumerate(datos_ocr['splits'][:6]):
                with cols[i % len(cols)]:
                    st.write(f"**{sp}**")

        with st.expander("Texto OCR (depuración)"):
            st.text_area("Header OCR", datos_ocr.get('header_text',''), height=140)
            st.text_area("Sports OCR", datos_ocr.get('sports_text',''), height=140)
            st.write("Filas SPLITS (parseadas):")
            st.json(datos_ocr.get('splits_rows', []))

    except Exception as e:
        st.error(f"Error en OCR: {e}")

# -------------------------------------------------------------------
# TrainingPeaks
# -------------------------------------------------------------------
df_tp = None
if tp_file:
    try:
        if tp_file.name.endswith('.csv'):
            df_tp = pd.read_csv(tp_file)
        else:
            df_tp = pd.read_excel(tp_file)
        st.success("TrainingPeaks cargado correctamente.")
    except Exception as e:
        st.error(f"Error leyendo TrainingPeaks: {e}")

# -------------------------------------------------------------------
# Procesamiento FIT y análisis
# -------------------------------------------------------------------
if fit_file is not None:
    try:
        with st.spinner("Analizando datos del ciclocomputador..."):
            df_fit = read_fit(fit_file)

        if df_fit is not None and not df_fit.empty:
            st.success("✅ Datos del ciclocomputador procesados correctamente")
            st.sidebar.info(f"**Puntos de datos:** {len(df_fit):,}")
            if 'timestamp' in df_fit.columns and len(df_fit) > 0:
                duracion_total = (df_fit['timestamp'].iloc[-1] - df_fit['timestamp'].iloc[0]).total_seconds() / 60
                st.sidebar.info(f"**Duración total:** {duracion_total:.1f} min")

            # Ajustar tiempos con OCR (si disponible)
            msg_aj = "No se ajustaron los tiempos (sin información de OCR)"
            if datos_ocr and datos_ocr.get('hora_inicio_real'):
                df_fit, msg_aj = ajustar_tiempos_con_ocr(df_fit, datos_ocr)
            st.info(f"🕒 {msg_aj}")

            # Normalización de columnas numéricas
            df_fit.columns = [c.lower() for c in df_fit.columns]
            for col in ['power','heart_rate','speed','altitude','cadence']:
                if col in df_fit.columns:
                    df_fit[col] = pd.to_numeric(df_fit[col], errors='coerce')

            # Segmentación
            with st.spinner("Aplicando segmentación por altimetría..."):
                df_fit['segment'] = smart_segmentation(
                    df_fit,
                    min_segment_seconds=min_segment_seconds,
                    min_elevation_gain=min_elevation_gain,
                    grade_threshold=grade_threshold
                )
            if 'power' in df_fit.columns:
                df_fit['wkg'] = df_fit['power'] / weight

            seg_summary = summarize_segments(df_fit, weight_kg=weight)

            st.sidebar.success(f"**Segmentos detectados:** {len(seg_summary)}")
            if not seg_summary.empty:
                tipos_segmentos = seg_summary['tipo'].value_counts()
                for tipo, count in tipos_segmentos.items():
                    st.sidebar.info(f"**{tipo}:** {count} segmentos")

            # Resumen general
            st.markdown("## 📊 Resumen General")
            summary = {}
            if 'timestamp' in df_fit.columns and len(df_fit) > 0:
                start_time = df_fit['timestamp'].iloc[0]
                end_time = df_fit['timestamp'].iloc[-1]
                duration = end_time - start_time
                summary['Hora de inicio'] = datos_ocr.get('hora_inicio_real', start_time.strftime('%H:%M:%S'))
                summary['Duración total'] = f"{duration.total_seconds()/60:.1f} min"
                if datos_ocr.get('tiempo_total'):
                    summary['Tiempo oficial'] = datos_ocr['tiempo_total']

            if 'power' in df_fit.columns:
                p_avg = float(df_fit['power'].dropna().mean()) if not df_fit['power'].dropna().empty else np.nan
                p_max = float(df_fit['power'].dropna().max()) if not df_fit['power'].dropna().empty else np.nan
                np_global = compute_np(df_fit['power'].fillna(0), df_fit['timestamp'])
                summary['Potencia promedio (W)'] = round(p_avg,1) if not np.isnan(p_avg) else None
                summary['Potencia máxima (W)'] = round(p_max,1) if not np.isnan(p_max) else None
                summary['Potencia Normalizada (NP)'] = round(np_global,1)
                if ftp_input > 0:
                    summary['IF'] = round(np_global/ftp_input, 3)
                    tss_est = (duration.total_seconds() * np_global * (np_global/ftp_input)) / (ftp_input * 3600) * 100
                    summary['TSS estimado'] = round(tss_est, 1)

            if 'heart_rate' in df_fit.columns:
                hr_avg = float(df_fit['heart_rate'].dropna().mean()) if not df_fit['heart_rate'].dropna().empty else np.nan
                hr_max = float(df_fit['heart_rate'].dropna().max()) if not df_fit['heart_rate'].dropna().empty else np.nan
                summary['FC promedio (bpm)'] = round(hr_avg,1) if not np.isnan(hr_avg) else None
                summary['FC máxima (bpm)'] = round(hr_max,1) if not np.isnan(hr_max) else None

            if 'speed' in df_fit.columns:
                v_avg = df_fit['speed'].dropna().mean()*3.6 if not df_fit['speed'].dropna().empty else np.nan
                v_max = df_fit['speed'].dropna().max()*3.6 if not df_fit['speed'].dropna().empty else np.nan
                summary['Velocidad promedio (km/h)'] = round(v_avg,1) if not np.isnan(v_avg) else None
                summary['Velocidad máxima (km/h)'] = round(v_max,1) if not np.isnan(v_max) else None

            if 'altitude' in df_fit.columns:
                elev_gain = df_fit['altitude'].diff().clip(lower=0).sum()
                summary['Desnivel positivo (m)'] = round(elev_gain, 0)
                summary['Altitud máxima (m)'] = round(float(df_fit['altitude'].dropna().max()),1) if not df_fit['altitude'].dropna().empty else None

            if datos_ocr.get('posicion'):
                summary['Posición oficial'] = f"{datos_ocr['posicion']}°"

            st.table(pd.DataFrame([(k, v) for k,v in summary.items() if v is not None], columns=['Métrica','Valor']))

            # Análisis integrado OCR
            if datos_ocr:
                st.markdown("## 🎯 ANÁLISIS INTEGRADO CON DATOS OFICIALES")
                st.markdown(generar_analisis_integrado_ocr(seg_summary, df_fit, ftp_input, weight, datos_ocr))

            # Segmentos
            st.markdown("## 🏔️ Análisis por Segmentos")
            if not seg_summary.empty:
                st.dataframe(seg_summary, use_container_width=True)
                cL, cR = st.columns(2)
                with cL:
                    st.markdown("### 💥 Más Intensos (NP W/kg)")
                    intensos = seg_summary.nlargest(5, 'np_wkg')[['tipo','duration_min','np_wkg','power_np_w','speed_avg_kmh']]
                    st.dataframe(intensos, use_container_width=True)
                with cR:
                    st.markdown("### ⏱️ Más Largos")
                    largos = seg_summary.nlargest(5, 'duration_min')[['tipo','duration_min','dist_m','np_wkg','speed_avg_kmh']].copy()
                    largos['dist_km'] = (largos['dist_m']/1000).round(2)
                    st.dataframe(largos[['tipo','duration_min','dist_km','np_wkg','speed_avg_kmh']], use_container_width=True)
            else:
                st.info("No se detectaron segmentos significativos con los parámetros actuales.")

            # Gráficas
            st.markdown("## 📈 Evolución Temporal")
            if 'timestamp' in df_fit.columns and len(df_fit) > 0:
                t0 = df_fit['timestamp'].iloc[0]
                df_fit['tiempo_transcurrido_min'] = (df_fit['timestamp'] - t0).dt.total_seconds()/60
                st.info(f"**Hora de inicio**: {datos_ocr.get('hora_inicio_real', t0.strftime('%H:%M:%S'))}")

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Evolución del Rendimiento', fontsize=16)

            if 'power' in df_fit.columns:
                axes[0,0].plot(df_fit['tiempo_transcurrido_min'], df_fit['power'], linewidth=1)
                axes[0,0].set_title('Potencia (W)'); axes[0,0].set_xlabel('Min'); axes[0,0].set_ylabel('W'); axes[0,0].grid(True, alpha=0.3)
                if not df_fit['power'].dropna().empty:
                    axes[0,0].axhline(df_fit['power'].mean(), linestyle='--', label='Prom', linewidth=1)
                    axes[0,0].legend()

            if 'heart_rate' in df_fit.columns:
                axes[0,1].plot(df_fit['tiempo_transcurrido_min'], df_fit['heart_rate'], linewidth=1)
                axes[0,1].set_title('Frecuencia Cardíaca (bpm)'); axes[0,1].set_xlabel('Min'); axes[0,1].set_ylabel('bpm'); axes[0,1].grid(True, alpha=0.3)

            if 'speed' in df_fit.columns:
                axes[1,0].plot(df_fit['tiempo_transcurrido_min'], df_fit['speed']*3.6, linewidth=1)
                axes[1,0].set_title('Velocidad (km/h)'); axes[1,0].set_xlabel('Min'); axes[1,0].set_ylabel('km/h'); axes[1,0].grid(True, alpha=0.3)

            if 'altitude' in df_fit.columns:
                axes[1,1].plot(df_fit['tiempo_transcurrido_min'], df_fit['altitude'], linewidth=1)
                axes[1,1].set_title('Altitud (m)'); axes[1,1].set_xlabel('Min'); axes[1,1].set_ylabel('m'); axes[1,1].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

            # Potencia por tipo de terreno
            if all(c in df_fit.columns for c in ['power','tiempo_transcurrido_min','segment']):
                st.markdown("### 🔴🟢🔵 Potencia por Terreno")
                fig2, ax = plt.subplots(figsize=(14,5))
                color_map = {'Subida':'red','Llano':'green','Bajada':'blue'}
                for t in ['Subida','Llano','Bajada']:
                    mask = df_fit['segment']==t
                    if mask.any():
                        ax.scatter(df_fit.loc[mask,'tiempo_transcurrido_min'],
                                   df_fit.loc[mask,'power'],
                                   s=8, alpha=0.6, label=t)
                ax.set_xlabel('Min'); ax.set_ylabel('W'); ax.grid(True, alpha=0.3); ax.legend()
                st.pyplot(fig2)

            # Reporte descargable
            st.markdown("## 💾 Reporte Completo (CSV)")
            buf = io.StringIO()
            buf.write("ANALISIS PROFESIONAL DE CARRERA\n")
            buf.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            buf.write(f"Pendiente umbral: {grade_threshold:.1f}%\n")
            if datos_ocr.get('hora_inicio_real'):
                buf.write(f"Hora oficial inicio: {datos_ocr['hora_inicio_real']}\n\n")
            else:
                buf.write("\n")
            buf.write("RESUMEN GENERAL\n")
            pd.DataFrame([(k,v) for k,v in summary.items() if v is not None], columns=['Métrica','Valor']).to_csv(buf, index=False)
            buf.write("\n\n")
            if not seg_summary.empty:
                buf.write("ANALISIS POR SEGMENTOS\n")
                seg_summary.to_csv(buf, index=False)
                buf.write("\n\n")
            st.download_button("📥 Descargar CSV", data=buf.getvalue(),
                               file_name=f"analisis_carrera_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                               mime="text/csv")
            st.success("🎉 Análisis profesional completado.")

        else:
            st.error("El archivo .FIT está vacío o no contiene datos válidos")

    except Exception as e:
        st.error(f"Error procesando archivo FIT: {str(e)}")
else:
    st.info("👆 Carga un archivo .FIT para comenzar el análisis")

st.markdown("---")
st.markdown("### 💡 Interpretación del Análisis")
st.info("""
- **Segmentos de Mayor Intensidad**: Momentos con mayor NP y W/kg.
- **Segmentos por Duración**: Tramos más largos e influyentes.
- **Evolución Temporal**: Variación de potencia, FC, velocidad y altitud.
- **Por Terreno**: Eficiencia en subidas, llanos y bajadas para orientar el entrenamiento.
""")