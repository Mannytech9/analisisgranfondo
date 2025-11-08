# app.py — Análisis Profesional de Ciclismo 2025
# Versión SIMPLIFICADA y MEJORADA

import os
import io
import re
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
from fitparse import FitFile
from datetime import datetime, timedelta, time
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="Análisis Profesional Ciclismo",
    layout="wide",
)

st.title("🚴‍♂️ Análisis Profesional de Rendimiento Ciclista")

# =============================================================================
# PARÁMETROS SIMPLIFICADOS
# =============================================================================
st.sidebar.header("⚙️ Parámetros del ciclista")
weight = st.sidebar.number_input("Peso ciclista (kg)", 50.0, 120.0, 70.0, 0.5)
ftp = st.sidebar.number_input("FTP (W)", 150, 500, 250, 10)

# Parámetros fijos para segmentación - ya no se piden al usuario
MIN_SEGMENT_SECONDS = 180  # 3 minutos mínimo por segmento
MIN_ELEVATION_GAIN = 25    # 25 metros mínimo de desnivel
GRADE_THRESHOLD = 3.0      # 3% de pendiente mínima para considerar subida

# =============================================================================
# CARGA DE ARCHIVOS
# =============================================================================
st.header("📁 Carga de archivos")

st.info("""
**Instrucciones:**
1. **Archivo .FIT** (obligatorio) - Datos de tu ciclocomputador
2. **Imagen de resultados** (opcional) - Foto del ranking oficial
""")

col1, col2 = st.columns(2)
with col1:
    uploaded_image = st.file_uploader("Imagen de resultados", type=["png", "jpg", "jpeg"])
with col2:
    fit_file = st.file_uploader("Archivo .FIT", type=["fit"])

# =============================================================================
# OCR MEJORADO - FUNCIONES CORREGIDAS
# =============================================================================

def normalize_image(img: Image.Image, target_h=2000):
    """Reescala manteniendo proporción"""
    w, h = img.size
    scale = target_h / h
    new_w = int(w * scale)
    img = img.resize((new_w, target_h), Image.LANCZOS)
    return img

def preprocess_for_ocr(bgr):
    """Preprocesamiento robusto para OCR"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # CLAHE para contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Desenfoque suave + unsharp masking
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    
    # Umbral adaptativo
    thr = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 41, 15
    )
    
    return thr

def extract_ocr_mejorado(full_img):
    """Extracción OCR completa y mejorada"""
    try:
        # Normalizar imagen
        img_norm = normalize_image(full_img, target_h=2200)
        bgr = cv2.cvtColor(np.array(img_norm), cv2.COLOR_RGB2BGR)
        
        # Preprocesar imagen completa
        full_bin = preprocess_for_ocr(bgr)
        H, W = full_bin.shape
        
        # Dividir en 3 secciones fijas (más simple y robusto)
        header = bgr[0:int(H*0.3), :]
        sports = bgr[int(H*0.3):int(H*0.6), :]
        splits = bgr[int(H*0.6):H, :]
        
        # Preprocesar cada sección
        hbin = preprocess_for_ocr(header)
        sbin = preprocess_for_ocr(sports)
        tbin = preprocess_for_ocr(splits)
        
        # OCR
        
        # Parsear datos básicos
        
        return resultado, (hbin, sbin, tbin)
        
    except Exception as e:
        st.error(f"Error en OCR: {str(e)}")
        return {}, (None, None, None)

    """Parser simplificado para datos OCR"""
    # Expresiones regulares
    rx_time_hms = re.compile(r"\b(\d{1,2}:\d{2}:\d{2})\b")
    rx_time_ms = re.compile(r"\b(\d{1,2}:\d{2})\b")
    rx_place = re.compile(r"\b(\d{1,4})\s*°")
    rx_speed = re.compile(r"(\d{1,2}[.,]\d{1,2})\s*km/h", re.IGNORECASE)
    rx_hora = re.compile(r"\b(\d{1,2}:\d{2})\b")
    
    resultado = {
        "posicion": None,
        "tiempo_total": None,
        "ritmo_promedio": None,
        "hora_inicio_real": None,
        "splits": []
    }
    
    # Buscar posición
    if m:
        resultado["posicion"] = m.group(1)
    
    # Buscar tiempo total (el HH:MM:SS más largo)
    if tiempos:
        # Encontrar el tiempo más largo
        max_seconds = 0
        best_time = None
        for tiempo in tiempos:
            h, m, s = map(int, tiempo.split(':'))
            total_seconds = h*3600 + m*60 + s
            if total_seconds > max_seconds:
                max_seconds = total_seconds
                best_time = tiempo
        resultado["tiempo_total"] = best_time
    
    # Buscar ritmo
    if m:
        resultado["ritmo_promedio"] = m.group(1).replace(',', '.')
    
    # Buscar hora de inicio (primer HH:MM que parece hora razonable)
    for hora in horas:
        h, m = map(int, hora.split(':'))
        if 6 <= h <= 12:  # Horas razonables para una carrera
            resultado["hora_inicio_real"] = hora + ":00"
            break
    
    # Extraer splits simples
        try:
            splits_times = rx_time_hms.findall(splits_text)
            resultado["splits"] = splits_times[:10]  # Máximo 10 splits
        except:
            pass
    
    return resultado

# =============================================================================
# PROCESAMIENTO FIT MEJORADO
# =============================================================================

def read_fit(file):
    """Lee archivo FIT"""
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
    """Calcula Potencia Normalizada"""
    try:
        p = pd.Series(series_power).fillna(0).astype(float)
        if len(timestamps) < 2:
            return float(p.mean())
        dt = np.median(np.diff(np.array(timestamps).astype('int64') // 1_000_000_000))
        window = max(1, int(round(30.0 / dt))) if dt > 0 else 1
        mov = p.rolling(window=window, min_periods=1).mean()
        np_val = (np.mean(mov**4)) ** 0.25
        return float(np_val)
    except:
        return float(series_power.mean())

def compute_grade_percent(df):
    """Calcula pendiente de forma robusta"""
    if 'altitude' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    d2 = df.copy()
    d2['alt_diff'] = d2['altitude'].diff().fillna(0)
    
    # Usar distancia si está disponible
    dist_col = None
    for c in d2.columns:
        if 'distance' in c.lower():
            dist_col = c
            break
    
    if dist_col and dist_col in d2.columns:
        d2['dist_diff'] = d2[dist_col].diff().replace(0, 0.1)  # Evitar división por cero
        grade = (d2['alt_diff'] / d2['dist_diff']).fillna(0) * 100
    else:
        # Fallback: estimar distancia con velocidad
        if 'speed' in d2.columns and 'timestamp' in d2.columns:
            d2['dt'] = d2['timestamp'].diff().dt.total_seconds().replace(0, 1)
            d2['dist_est'] = d2['speed'] * d2['dt']
            grade = (d2['alt_diff'] / d2['dist_est']).fillna(0) * 100
        else:
            grade = pd.Series(0.0, index=d2.index)
    
    return grade.replace([np.inf, -np.inf], 0).fillna(0)

def smart_segmentation(df):
    """Segmentación inteligente con parámetros fijos"""
    if 'altitude' not in df.columns or 'timestamp' not in df.columns:
        return pd.Series(['Llano'] * len(df), index=df.index)
    
    d = df.copy()
    
    # Calcular pendiente
    d['grade_pct'] = compute_grade_percent(d)
    
    # Suavizar pendiente para evitar cambios bruscos
    d['grade_smooth'] = d['grade_pct'].rolling(window=10, center=True, min_periods=1).mean()
    
    # Identificar subidas (pendiente > 3% por al menos 3 minutos)
    climb_mask = (d['grade_smooth'] > GRADE_THRESHOLD)
    
    # Agrupar segmentos de subida
    segments = []
    current_segment = None
    
    for i in range(len(d)):
        if climb_mask.iloc[i] and current_segment is None:
            current_segment = {'start': i, 'type': 'Subida'}
        elif not climb_mask.iloc[i] and current_segment is not None:
            current_segment['end'] = i
            # Verificar si el segmento cumple con los requisitos mínimos
            duration = (d['timestamp'].iloc[current_segment['end']] - d['timestamp'].iloc[current_segment['start']]).total_seconds()
            elevation_gain = d['altitude'].iloc[current_segment['end']] - d['altitude'].iloc[current_segment['start']]
            
            if duration >= MIN_SEGMENT_SECONDS and elevation_gain >= MIN_ELEVATION_GAIN:
                segments.append(current_segment)
            current_segment = None
    
    # Si terminamos con un segmento activo
    if current_segment is not None:
        current_segment['end'] = len(d) - 1
        duration = (d['timestamp'].iloc[current_segment['end']] - d['timestamp'].iloc[current_segment['start']]).total_seconds()
        elevation_gain = d['altitude'].iloc[current_segment['end']] - d['altitude'].iloc[current_segment['start']]
        
        if duration >= MIN_SEGMENT_SECONDS and elevation_gain >= MIN_ELEVATION_GAIN:
            segments.append(current_segment)
    
    # Crear etiquetas
    labels = ['Llano'] * len(d)
    
    for seg in segments:
        labels[seg['start']:seg['end']+1] = ['Subida'] * (seg['end'] - seg['start'] + 1)
    
    # Identificar bajadas significativas
    for i in range(len(d)):
        if labels[i] == 'Llano' and d['grade_smooth'].iloc[i] < -GRADE_THRESHOLD:
            labels[i] = 'Bajada'
    
    return pd.Series(labels, index=df.index)

def summarize_segments(df, weight_kg):
    """Resume segmentos - versión mejorada"""
    if 'segment' not in df.columns or 'timestamp' not in df.columns or len(df) == 0:
        return pd.DataFrame()
    
    d = df.sort_values('timestamp').reset_index(drop=True)
    d['seg_block'] = (d['segment'] != d['segment'].shift(1)).cumsum()
    rows = []
    
    for sid, g in d.groupby('seg_block'):
        seg_type = g['segment'].iloc[0]
        t0, t1 = g['timestamp'].iloc[0], g['timestamp'].iloc[-1]
        duration_min = (t1 - t0).total_seconds() / 60.0
        
        # Solo considerar segmentos de al menos 2 minutos
        if duration_min < 2.0:
            continue
            
        # Distancia
        dist_m = 0.0
        for c in g.columns:
            if 'distance' in c.lower():
                try:
                    dist_m = float(g[c].iloc[-1] - g[c].iloc[0])
                    break
                except:
                    pass
        
        # Elevación
        elevation_gain = 0.0
        if 'altitude' in g.columns and seg_type == 'Subida':
            elevation_gain = float(g['altitude'].iloc[-1] - g['altitude'].iloc[0])
        
        # Potencia
        avg_power = float(g['power'].dropna().mean()) if 'power' in g.columns and not g['power'].dropna().empty else np.nan
        max_power = float(g['power'].dropna().max()) if 'power' in g.columns and not g['power'].dropna().empty else np.nan
        np_val = compute_np(g['power'].fillna(0), g['timestamp']) if 'power' in g.columns else np.nan
        
        # FC
        avg_hr = float(g['heart_rate'].dropna().mean()) if 'heart_rate' in g.columns and not g['heart_rate'].dropna().empty else np.nan
        
        # Velocidad
        avg_speed_kmh = float(g['speed'].dropna().mean()*3.6) if 'speed' in g.columns and not g['speed'].dropna().empty else np.nan
        
        # W/kg
        avg_wkg = avg_power / weight_kg if not np.isnan(avg_power) else np.nan
        np_wkg = np_val / weight_kg if not np.isnan(np_val) else np.nan
        
        rows.append({
            'segment_id': int(sid),
            'tipo': seg_type,
            'duracion_min': round(duration_min, 1),
            'distancia_km': round(dist_m/1000, 2),
            'desnivel_m': round(elevation_gain, 0),
            'potencia_promedio_w': round(avg_power,0) if not np.isnan(avg_power) else np.nan,
            'potencia_maxima_w': round(max_power,0) if not np.isnan(max_power) else np.nan,
            'potencia_normalizada_w': round(np_val,0) if not np.isnan(np_val) else np.nan,
            'np_wkg': round(np_wkg,2) if not np.isnan(np_wkg) else np.nan,
            'velocidad_promedio_kmh': round(avg_speed_kmh,1) if not np.isnan(avg_speed_kmh) else np.nan,
            'fc_promedio': round(avg_hr,0) if not np.isnan(avg_hr) else np.nan
        })
    
    return pd.DataFrame(rows)

# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================

# Procesar imagen OCR
datos_ocr = {}
if uploaded_image:
    try:
        st.markdown("## 📷 Imagen de Resultados")
        im = Image.open(uploaded_image).convert('RGB')
        show = im.copy()
        show.thumbnail((700, 700))
        st.image(show, caption="Resultados oficiales", use_container_width=False)

        with st.spinner("Procesando imagen (OCR Inteligente)…"):
            datos_ocr, (hbin, sbin, tbin) = extract_ocr_mejorado(im)

        if datos_ocr:
            # Mostrar imágenes binarizadas si están disponibles
            if hbin is not None:
                colA, colB, colC = st.columns(3)
                with colA:
                    st.image(hbin, caption="Header binarizado", use_container_width=True, clamp=True)
                with colB:
                    st.image(sbin, caption="Sports binarizado", use_container_width=True, clamp=True)
                with colC:
                    st.image(tbin, caption="Splits binarizado", use_container_width=True, clamp=True)

            # Información extraída
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

            # Splits
            if datos_ocr.get('splits'):
                st.markdown("#### ⏱️ Splits (tiempo de carrera)")
                cols = st.columns(min(6, len(datos_ocr['splits'])))
                for i, sp in enumerate(datos_ocr['splits'][:6]):
                    with cols[i % len(cols)]:
                        st.write(f"**{sp}**")

    except Exception as e:
        st.error(f"❌ Error en OCR: {e}")

# Procesar archivo FIT
if fit_file:
    try:
        with st.spinner("Analizando datos del ciclocomputador..."):
            df_fit = read_fit(fit_file)

        if df_fit is not None and not df_fit.empty:
            st.success(f"✅ Datos del ciclocomputador procesados correctamente ({len(df_fit)} puntos)")
            
            # Normalizar columnas
            df_fit.columns = [c.lower() for c in df_fit.columns]
            for col in ['power','heart_rate','speed','altitude','cadence']:
                if col in df_fit.columns:
                    df_fit[col] = pd.to_numeric(df_fit[col], errors='coerce')

            # Segmentación automática
            with st.spinner("Aplicando segmentación automática..."):
                df_fit['segment'] = smart_segmentation(df_fit)
            
            if 'power' in df_fit.columns:
                df_fit['wkg'] = df_fit['power'] / weight

            seg_summary = summarize_segments(df_fit, weight_kg=weight)

            # Resumen general
            st.markdown("## 📊 Resumen General")
            summary_data = []
            
            if 'timestamp' in df_fit.columns and len(df_fit) > 0:
                start_time = df_fit['timestamp'].iloc[0]
                end_time = df_fit['timestamp'].iloc[-1]
                duration = end_time - start_time
                total_minutes = duration.total_seconds() / 60
                
                summary_data.append(["Duración total", f"{total_minutes:.1f} min"])
                summary_data.append(["Hora de inicio", datos_ocr.get('hora_inicio_real', start_time.strftime('%H:%M:%S'))])
                
                if datos_ocr.get('tiempo_total'):
                    summary_data.append(["Tiempo oficial", datos_ocr['tiempo_total']])

            # Métricas de potencia
            if 'power' in df_fit.columns:
                p_avg = float(df_fit['power'].dropna().mean()) if not df_fit['power'].dropna().empty else np.nan
                p_max = float(df_fit['power'].dropna().max()) if not df_fit['power'].dropna().empty else np.nan
                np_global = compute_np(df_fit['power'].fillna(0), df_fit['timestamp'])
                
                if not np.isnan(p_avg):
                    summary_data.append(["Potencia promedio", f"{p_avg:.0f} W"])
                    summary_data.append(["Potencia máxima", f"{p_max:.0f} W"])
                    summary_data.append(["Potencia Normalizada (NP)", f"{np_global:.0f} W"])
                    summary_data.append(["W/kg promedio", f"{p_avg/weight:.2f}"])
                    
                    if ftp > 0:
                        IF = np_global / ftp
                        TSS = (total_minutes / 60) * (np_global * IF) / ftp * 100
                        summary_data.append(["Intensidad (IF)", f"{IF:.2f}"])
                        summary_data.append(["TSS estimado", f"{TSS:.0f}"])

            # Velocidad
            if 'speed' in df_fit.columns:
                v_avg = df_fit['speed'].dropna().mean()*3.6 if not df_fit['speed'].dropna().empty else np.nan
                if not np.isnan(v_avg):
                    summary_data.append(["Velocidad promedio", f"{v_avg:.1f} km/h"])

            # FC
            if 'heart_rate' in df_fit.columns:
                hr_avg = df_fit['heart_rate'].dropna().mean() if not df_fit['heart_rate'].dropna().empty else np.nan
                hr_max = df_fit['heart_rate'].dropna().max() if not df_fit['heart_rate'].dropna().empty else np.nan
                if not np.isnan(hr_avg):
                    summary_data.append(["FC promedio", f"{hr_avg:.0f} bpm"])
                    summary_data.append(["FC máxima", f"{hr_max:.0f} bpm"])

            # Altitud
            if 'altitude' in df_fit.columns:
                elev_gain = df_fit['altitude'].diff().clip(lower=0).sum()
                elev_max = df_fit['altitude'].max() if not df_fit['altitude'].dropna().empty else np.nan
                summary_data.append(["Desnivel positivo", f"{elev_gain:.0f} m"])
                if not np.isnan(elev_max):
                    summary_data.append(["Altitud máxima", f"{elev_max:.0f} m"])

            # Posición oficial
            if datos_ocr.get('posicion'):
                summary_data.append(["Posición oficial", f"{datos_ocr['posicion']}°"])

            # Mostrar tabla de resumen
            st.table(pd.DataFrame(summary_data, columns=['Métrica', 'Valor']))

            # Segmentos (solo si hay segmentos significativos)
            if not seg_summary.empty and len(seg_summary) > 0:
                st.markdown("## 🏔️ Análisis por Segmentos")
                
                # Filtrar solo segmentos más relevantes
                relevant_segments = seg_summary[seg_summary['duracion_min'] >= 2.0]
                
                if len(relevant_segments) > 0:
                    st.dataframe(relevant_segments, use_container_width=True)
                    
                    # Mostrar segmentos más intensos
                    if 'np_wkg' in relevant_segments.columns:
                        intensos = relevant_segments.nlargest(3, 'np_wkg')
                        if len(intensos) > 0:
                            st.markdown("### 💥 Segmentos Más Intensos")
                            st.dataframe(intensos[['tipo', 'duracion_min', 'np_wkg', 'potencia_normalizada_w']], 
                                       use_container_width=True)
                else:
                    st.info("No se detectaron segmentos significativos (> 2 minutos)")

            # Gráficas
            st.markdown("## 📈 Evolución Temporal")
            if 'timestamp' in df_fit.columns and len(df_fit) > 0:
                t0 = df_fit['timestamp'].iloc[0]
                df_fit['minutos_desde_inicio'] = (df_fit['timestamp'] - t0).dt.total_seconds()/60

                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('Evolución del Rendimiento', fontsize=16)

                # Potencia
                if 'power' in df_fit.columns:
                    axes[0,0].plot(df_fit['minutos_desde_inicio'], df_fit['power'], linewidth=1, color='red', alpha=0.7)
                    axes[0,0].set_title('Potencia (W)')
                    axes[0,0].set_xlabel('Minutos')
                    axes[0,0].grid(True, alpha=0.3)
                    if not df_fit['power'].dropna().empty:
                        axes[0,0].axhline(df_fit['power'].mean(), linestyle='--', color='red', alpha=0.8, label=f'Prom: {df_fit["power"].mean():.0f}W')
                        axes[0,0].legend()

                # FC
                if 'heart_rate' in df_fit.columns:
                    axes[0,1].plot(df_fit['minutos_desde_inicio'], df_fit['heart_rate'], linewidth=1, color='green', alpha=0.7)
                    axes[0,1].set_title('Frecuencia Cardíaca (bpm)')
                    axes[0,1].set_xlabel('Minutos')
                    axes[0,1].grid(True, alpha=0.3)

                # Velocidad
                if 'speed' in df_fit.columns:
                    axes[1,0].plot(df_fit['minutos_desde_inicio'], df_fit['speed']*3.6, linewidth=1, color='blue', alpha=0.7)
                    axes[1,0].set_title('Velocidad (km/h)')
                    axes[1,0].set_xlabel('Minutos')
                    axes[1,0].grid(True, alpha=0.3)

                # Altitud
                if 'altitude' in df_fit.columns:
                    axes[1,1].plot(df_fit['minutos_desde_inicio'], df_fit['altitude'], linewidth=1, color='brown', alpha=0.7)
                    axes[1,1].set_title('Altitud (m)')
                    axes[1,1].set_xlabel('Minutos')
                    axes[1,1].grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

            st.success("🎉 Análisis profesional completado.")

    except Exception as e:
        st.error(f"Error procesando archivo FIT: {str(e)}")

# Mensaje final
if not fit_file:
    st.info("👆 Carga un archivo .FIT para comenzar el análisis")

st.markdown("---")
st.markdown("### 💡 Acerca del Análisis")
st.info("""
- **Segmentación Automática**: Los segmentos se detectan automáticamente basándose en cambios de pendiente (>3%) y desnivel (>25m)
- **Métricas Clave**: NP (Potencia Normalizada), IF (Factor de Intensidad), TSS (Training Stress Score)
- **Datos Oficiales**: La información del OCR se usa para comparar con los datos del ciclocomputador
""")


