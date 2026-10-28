import os
import base64
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuración de página
st.set_page_config(
    page_title="AI PhotoRestorer Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ESTILOS CSS PROFESIONALES Y DISEÑO A MEDIDA ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0b0f17;
    }

    /* Título principal con estilo único y borde resaltado abajo */
    .hero-container {
        padding: 1.5rem 0 2rem 0;
        border-bottom: 2px solid #1e293b;
        margin-bottom: 2rem;
    }

    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.5px;
        margin: 0;
        background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 8px;
        font-weight: 400;
    }

    /* Badge compacto del backend */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.8);
        border: 1px solid #334155;
        color: #e2e8f0;
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }
    .status-dot.online { background-color: #22c55e; box-shadow: 0 0 8px #22c55e; }
    .status-dot.offline { background-color: #ef4444; box-shadow: 0 0 8px #ef4444; }

    /* Estilos de Pestañas Principales */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        background-color: transparent;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        padding: 0 24px;
        border-radius: 8px;
        font-weight: 600;
        color: #64748b;
        background-color: #111827;
        border: 1px solid #1f2937;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        border: none !important;
    }

    /* Botón de acción principal */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4);
        width: 100%;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6);
    }

    /* Tarjetas de estadísticas */
    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1f2937;
        padding: 16px;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

API = os.getenv('API_URL', 'http://127.0.0.1:8000')

# Verificación de conexión backend silenciosa
backend_online = False
backend_msg = "Backend Offline"
try:
    health_check = requests.get(f'{API}/health', timeout=2).json()
    if health_check.get('api') == 'ok':
        backend_online = True
        backend_msg = "Backend Funcional (Online)"
except Exception:
    backend_online = False

# --- ENCABEZADO Y TÍTULO CON ESTILO ---
col_head1, col_head2 = st.columns([5, 1])
with col_head1:
    st.markdown("""
        <div class="hero-container">
            <h1 class="hero-title">✨ AI PhotoRestorer Studio</h1>
            <p class="hero-subtitle">Plataforma inteligente de superresolución y restauración facial avanzada impulsada por Real-ESRGAN y CodeFormer para recuperar tus recuerdos con máxima nitidez.</p>
        </div>
    """, unsafe_allow_html=True)
with col_head2:
    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class="status-badge">
            <span class="status-dot {'online' if backend_online else 'offline'}"></span>
            {backend_msg}
        </div>
    """, unsafe_allow_html=True)

# --- MENÚ DE NAVEGACIÓN PRINCIPAL ---
tab_home, tab_history = st.tabs(['🚀 Restaurador Principal', '📂 Fotos Almacenadas & Estadísticas'])

with tab_home:
    # 1. Cargar imagen
    uploaded = st.file_uploader('Sube tu fotografía aquí', type=['png', 'jpg', 'jpeg', 'webp'], label_visibility="collapsed")
    
    if uploaded:
        raw_uploaded = uploaded.getvalue()
        
        # 2. Botón de mejora
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            trigger_enhance = st.button('🚀 Mejorar y Reconstruir Fotografía', type='primary')
            
        if trigger_enhance:
            try:
                with st.spinner('Procesando imagen con IA (Ampliación y restauración facial)...'):
                    r = requests.post(
                        f'{API}/restore',
                        files={'file': (uploaded.name, raw_uploaded, uploaded.type)},
                        data={'scale': 4.0},
                        timeout=600
                    )
                    r.raise_for_status()
                    d = r.json()

                output = base64.b64decode(d['image_base64'])
                st.session_state['last_restored'] = output
                st.session_state['last_original'] = raw_uploaded
                st.session_state['last_metrics'] = d.get('metrics', {})
                st.success(f"¡Proceso completado con éxito! Job ID: {d['id']} | Dispositivo: {d['device'].upper()}")
            except Exception as e:
                st.error(f"Error al procesar la imagen: {e}")

        # 3. Mostrar la foto mejorada vs la normal
        if 'last_restored' in st.session_state:
            st.markdown("---")
            st.markdown("### 🔍 Comparativa Visual")
            col_orig, col_rest = st.columns(2)
            with col_orig:
                st.image(st.session_state['last_original'], caption='Imagen Normal (Original)', use_container_width=True)
            with col_rest:
                st.image(st.session_state['last_restored'], caption='Imagen Mejorada (Reconstruida)', use_container_width=True)
            
            st.download_button('💾 Descargar Imagen Mejorada', st.session_state['last_restored'], 'foto_mejorada.png', 'image/png', use_container_width=True)

            # 4. Abajo: Estadísticas de la foto actual
            st.markdown("---")
            st.markdown("### 📊 Estadísticas de Calidad")
            metrics = st.session_state.get('last_metrics', {})
            if metrics:
                m1, m2, m3 = st.columns(3)
                m1.metric("Nitidez Inicial (Laplaciano)", metrics.get('laplacian_before', 0))
                m2.metric("Nitidez Mejorada (Laplaciano)", metrics.get('laplacian_after', 0))
                m3.metric("Fidelidad Estructural (PSNR)", f"{metrics.get('psnr', 0)} dB")
    else:
        st.info("👆 Por favor, carga una imagen en formato JPG o PNG para comenzar la restauración.")

with tab_history:
    st.markdown("### 📂 Archivo de Fotografías Almacenadas")
    try:
        r = requests.get(f'{API}/history', timeout=15)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data)
        
        if len(df) > 0:
            available_cols = [col for col in ['id', 'timestamp', 'created_at', 'device', 'seconds', 'psnr'] if col in df.columns]
            st.dataframe(df[available_cols], use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 📈 Gráficas y Análisis Global")
            c1, c2 = st.columns(2)
            with c1:
                if 'seconds' in df.columns:
                    st.plotly_chart(px.histogram(df, x='seconds', title='Tiempos de Inferencia (Segundos)', template="plotly_dark"), use_container_width=True)
            with c2:
                if 'device' in df.columns:
                    st.plotly_chart(px.histogram(df, x='device', title='Distribución por Dispositivo (CUDA / CPU)', template="plotly_dark"), use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 🔎 Inspeccionar Trabajo Almacenado")
            selected_id = st.selectbox('Selecciona el ID del registro:', df['id'])
            row_sel = df[df['id'] == selected_id].iloc[0]
            
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Nitidez Antes", row_sel.get('laplacian_before', 0))
            sc2.metric("Nitidez Después", row_sel.get('laplacian_after', 0))
            sc3.metric("PSNR", f"{row_sel.get('psnr', 0)} dB")
            
            img_o = requests.get(f'{API}/image/{selected_id}?type=original', timeout=30)
            img_r = requests.get(f'{API}/image/{selected_id}?type=restored', timeout=30)
            
            if img_o.status_code == 200 and img_r.status_code == 200:
                ic1, ic2 = st.columns(2)
                with ic1:
                    st.image(img_o.content, caption='Original Almacenada', use_container_width=True)
                with ic2:
                    st.image(img_r.content, caption='Mejorada Almacenada', use_container_width=True)
        else:
            st.warning("No hay registros almacenados en la base de datos todavía.")
    except Exception as e:
        st.error(f"No se pudo conectar con el servidor backend para listar el historial: {e}")