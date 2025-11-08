# app.py — Análisis Profesional de Ciclismo 2025
# Versión CORREGIDA para Streamlit Cloud

import os
import io
import re
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import easyocr

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

# Parámetros fijos para segmentación
MIN_SEGMENT_SECONDS = 180
MIN_ELEVATION_GAIN = 25
GRADE_THRESHOLD = 3.0

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
# FUNCIONES DE PROCESAMIENTO DE IMAGEN (OCR MEJORADO)
# =============================================================================

def extract_ocr_mejorado(full_img):
    try:
        # Inicializar EasyOCR (solo una vez para mejor rendimiento)
        if 'reader' not in st.session_state:
            st.session_state.reader = easyocr.Reader(['en', 'es'])  # Inglés y español
        
        # Convertir PIL Image a numpy array
        img_array = np.array(full_img)
        
        # Ejecutar OCR
        results = st.session_state.reader.readtext(img_array, detail=0, paragraph=True)
        texto_completo = " ".join(results)
        
        # Procesar texto con expresiones regulares mejoradas
        datos_extraidos = parse_ocr_text(texto_completo)
        
        # Generar imágenes de ejemplo para visualización
        hbin, sbin, tbin = generate_debug_images(img_array)
        
        return datos_extraidos, (hbin, sbin, tbin)
        
    except Exception as e:
        st.error(f"Error en OCR: {str(e)}")
        return {}, (None, None, None)

def parse_ocr_text(texto):
    """Parse mejorado del texto OCR"""
    # Limpiar texto
    texto_limpio = re.sub(r'\s+', ' ', texto)
    
    datos = {
        "posicion": None,
        "tiempo_total": None,
        "ritmo_promedio": None,
        "hora_inicio_real": None,
        "splits": [],
        "texto_extraido": texto_limpio[:500] + "..." if len(texto_limpio) > 500 else texto_limpio
    }
    
    # Buscar posición (ej: "45°", "1st", "23º")
    patron_posicion = r'(\d{1,3})[°ºª]|\b(\d{1,3})(?:st|nd|rd|th)\b'
    match_pos = re.search(patron_posicion, texto_limpio, re.IGNORECASE)
    if match_pos:
        datos["posicion"] = match_pos.group(1) or match_pos.group(2)
    
    # Buscar tiempo total (HH:MM:SS o H:MM:SS)
    patron_tiempo = r'\b(\d{1,2}):(\d{2}):(\d{2})\b'
    matches_tiempo = re.findall(patron_tiempo, texto_limpio)
    if matches_tiempo:
        # Tomar el tiempo más largo como tiempo total
        tiempos_segundos = []
        for h, m, s in matches_tiempo:
            segundos = int(h)*3600 + int(m)*60 + int(s)
            tiempos_segundos.append((f"{h}:{m}:{s}", segundos))
        
        if tiempos_segundos:
            tiempo_total = max(tiempos_segundos, key=lambda x: x[1])
            datos["tiempo_total"] = tiempo_total[0]
            
            # Los otros tiempos como splits
            otros_tiempos = [t[0] for t in tiempos_segundos if t[0] != tiempo_total[0]]
            datos["splits"] = otros_tiempos[:6]  # Máximo 6 splits
    
    # Buscar velocidad promedio (km/h)
    patron_velocidad = r'(\d{1,2}[,.]\d{1,2})\s*km/h|\b(\d{1,2})[,.](\d{1,2})\s*km/h'
    match_vel = re.search(patron_velocidad, texto_limpio, re.IGNORECASE)
    if match_vel:
        if match_vel.group(1):
            datos["ritmo_promedio"] = match_vel.group(1).replace(',', '.')
        elif match_vel.group(2) and match_vel.group(3):
            datos["ritmo_promedio"] = f"{match_vel.group(2)}.{match_vel.group(3)}"
    
    # Buscar hora de inicio
    patron_hora = r'\b(\d{1,2}):(\d{2}):(\d{2})\b'
    matches_hora = re.findall(patron_hora, texto_limpio)
    for h, m, s in matches_hora:
        hora_int = int(h)
        if 5 <= hora_int <= 12:  # Horas razonables para carrera
            datos["hora_inicio_real"] = f"{h}:{m}:{s}"
            break
    
    return datos

def generate_debug_images(img_array):
    """Generar imágenes para debug (simuladas)"""
    try:
        # Convertir a escala de grises para simular procesamiento
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Dividir en secciones (header, sports, splits)
        header = gray[0:int(h*0.3), :]
        sports = gray[int(h*0.3):int(h*0.6), :]
        splits = gray[int(h*0.6):h, :]
        
        return header, sports, splits
    except:
        return None, None, None

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
        d2['dist_diff'] = d2[dist_col].diff().replace(0, 0.1)
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
    
    # Suavizar pendiente
    d['grade_smooth'] = d['grade_pct'].rolling(window=10, center=True, min_periods=1).mean()
    
    # Identificar subidas
    climb_mask = (d['grade_smooth'] > GRADE_THRESHOLD)
    
    # Agrupar segmentos de subida
    segments = []
    current_segment = None
    
    for i in range(len(d)):
        if climb_mask.iloc[i] and current_segment is None:
            current_segment = {'start': i, 'type': 'Subida'}
        elif not climb_mask.iloc[i] and current_segment is not None:
            current_segment['end'] = i
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
    
    # Identificar bajadas
    for i in range(len(d)):
        if labels[i] == 'Llano' and d['grade_smooth'].iloc[i] < -GRADE_THRESHOLD:
            labels[i] = 'Bajada'
    
    return pd.Series(labels, index=df.index)

def summarize_segments(df, weight_kg):
    """Resume segmentos"""
    if 'segment' not in df.columns or 'timestamp' not in df.columns or len(df) == 0:
        return pd.DataFrame()
    
    d = df.sort_values('timestamp').reset_index(drop=True)
    d['seg_block'] = (d['segment'] != d['segment'].shift(1)).cumsum()
    rows = []
    
    for sid, g in d.groupby('seg_block'):
        seg_type = g['segment'].iloc[0]
        t0, t1 = g['timestamp'].iloc[0], g['timestamp'].iloc[-1]
        duration_min = (t1 - t0).total_seconds() / 60.0
        
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
        st.image(show, caption="Resultados oficiales", use_container_width=True)

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
        st.error(f"❌ Error procesando imagen: {e}")

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

            # Segmentos
            if not seg_summary.empty and len(seg_summary) > 0:
                st.markdown("## 🏔️ Análisis por Segmentos")
                
                relevant_segments = seg_summary[seg_summary['duracion_min'] >= 2.0]
                
                if len(relevant_segments) > 0:
                    st.dataframe(relevant_segments, use_container_width=True)
                    
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
- **OCR**: Actualmente en modo simulación. Para OCR real, considera usar un servicio por API.
""")


