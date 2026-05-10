import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
from datetime import datetime, date

# Configuración de página
st.set_page_config(page_title="Financial Sentiment Dashboard", layout="wide", page_icon="📈")

# Estilos básicos mejorados
st.markdown("""
<style>
    /* Estilos para las tarjetas de métricas */
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Centrar títulos de secciones */
    .st-emotion-cache-10trblm h1, .st-emotion-cache-10trblm h2, .st-emotion-cache-10trblm h3 {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Función para conectarse a la DB
@st.cache_resource
def get_connection():
    db_path = Path("data/sentiment.db")
    if not db_path.exists():
        st.error("No se encontró la base de datos `data/sentiment.db`. Asegúrate de haber ejecutado la ingesta y procesado de sentimiento.")
        st.stop()
    return sqlite3.connect(db_path, check_same_thread=False)

def get_data(query, params=()):
    conn = get_connection()
    return pd.read_sql(query, conn, params=params)

# ==========================================
# 1. KPIs Generales
# ==========================================
st.title("📈 Análisis de Sentimiento Financiero")
st.markdown("Dashboard interactivo para visualizar el sentimiento de empresas basado en menciones.")

# Consultas para KPIs
total_posts_df = get_data("SELECT COUNT(DISTINCT raw_document_id) as total FROM sentiment_results")
total_posts = total_posts_df.iloc[0]['total'] if not total_posts_df.empty else 0

global_avg_df = get_data("SELECT AVG(sentiment_score) as avg_score FROM sentiment_results")
global_avg = global_avg_df.iloc[0]['avg_score'] if not global_avg_df.empty else 0.0
if pd.isna(global_avg): global_avg = 0.0

# Empresa más mencionada hoy (usamos ingested_at si published_at no está o como aproximación)
mentions_today_query = """
SELECT c.name, COUNT(DISTINCT sr.raw_document_id) as count
FROM sentiment_results sr
JOIN companies c ON c.id = sr.company_id
JOIN raw_documents rd ON sr.raw_document_id = rd.id
WHERE date(rd.ingested_at) >= date('now', 'start of day') OR date(rd.published_at) >= date('now', 'start of day')
GROUP BY c.id
ORDER BY count DESC
LIMIT 1
"""
mentions_today_df = get_data(mentions_today_query)
top_mentioned_today = mentions_today_df.iloc[0]['name'] if not mentions_today_df.empty else "N/A"
top_mentioned_count = mentions_today_df.iloc[0]['count'] if not mentions_today_df.empty else 0

# Empresa más positiva y negativa (mínimo 3 posts para no tener sesgo de 1 solo post)
company_sentiment_query = """
SELECT c.name, AVG(sr.sentiment_score) as avg_score, COUNT(sr.raw_document_id) as posts
FROM sentiment_results sr
JOIN companies c ON c.id = sr.company_id
GROUP BY c.id
HAVING posts >= 2
ORDER BY avg_score DESC
"""
company_sentiment_df = get_data(company_sentiment_query)

if not company_sentiment_df.empty:
    most_positive = company_sentiment_df.iloc[0]['name']
    most_positive_score = company_sentiment_df.iloc[0]['avg_score']
    most_negative = company_sentiment_df.iloc[-1]['name']
    most_negative_score = company_sentiment_df.iloc[-1]['avg_score']
else:
    most_positive, most_positive_score = "N/A", 0.0
    most_negative, most_negative_score = "N/A", 0.0

st.header("Resumen General del Mercado")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Total Posts Analizados", value=f"{total_posts:,}")

with col2:
    st.metric(label="Sentimiento Global", value=f"{global_avg:.3f}")

with col3:
    label_mention = "Empresa más mencionada (Hoy)"
    st.metric(label=label_mention, value=top_mentioned_today, delta=f"{top_mentioned_count} posts" if top_mentioned_count else None, delta_color="off")

with col4:
    st.metric(label="Compañía más positiva", value=most_positive, delta=f"{most_positive_score:.2f}")

st.markdown("---")

# ==========================================
# 2. Ranking de Sentimiento
# ==========================================
st.markdown("<br><hr>", unsafe_allow_html=True)
st.subheader("🏆 Ranking de Sentimiento por Empresa")
ranking_query = """
SELECT c.name, c.ticker, AVG(sr.sentiment_score) as avg_score, COUNT(DISTINCT sr.raw_document_id) as mentions
FROM sentiment_results sr
JOIN companies c ON c.id = sr.company_id
GROUP BY c.id
HAVING mentions >= 2
ORDER BY avg_score DESC
LIMIT 15
"""
ranking_df = get_data(ranking_query)

if not ranking_df.empty:
    fig_ranking = px.bar(
        ranking_df, x='avg_score', y='name', orientation='h',
        labels={'avg_score': 'Sentimiento Promedio (-1 a 1)', 'name': ''},
        text=ranking_df['avg_score'].round(3),
        color='avg_score',
        color_continuous_scale="RdYlGn",
        range_x=[-1, 1]
    )
    fig_ranking.update_traces(textposition='outside')
    fig_ranking.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        height=500
    )
    st.plotly_chart(fig_ranking, use_container_width=True)
else:
    st.info("No hay suficientes datos para mostrar el ranking.")


st.markdown("<br><hr>", unsafe_allow_html=True)

# ==========================================
# 3. Selector de Empresa y Serie Temporal
# ==========================================
st.header("Análisis Detallado por Empresa")

companies_df = get_data("SELECT id, name, ticker FROM companies ORDER BY name")
if not companies_df.empty:
    companies_list = [f"{row['name']} ({row['ticker']})" for _, row in companies_df.iterrows()]
    
    selected_company_str = st.selectbox("Selecciona una empresa", options=companies_list)
    
    # Extraer el ticker
    selected_ticker = selected_company_str.split("(")[-1].replace(")", "")
    
    # Mostrar logo y nombre
    col_logo, col_title = st.columns([1, 11])
    with col_logo:
        # Usamos un servicio gratuito de logos de activos financieros (Parqet o UI Avatars como fallback)
        logo_url = f"https://assets.parqet.com/logos/symbol/{selected_ticker}?format=png"
        st.image(logo_url, width=64)
    with col_title:
        st.subheader(selected_company_str)
    
    company_id_df = get_data(f"SELECT id FROM companies WHERE ticker = '{selected_ticker}'")
    if not company_id_df.empty:
        selected_company_id = int(company_id_df.iloc[0]['id'])
        
        # Serie temporal
        time_series_query = """
        SELECT date(rd.published_at) as date, 
               AVG(sr.sentiment_score) as avg_sentiment, 
               COUNT(sr.raw_document_id) as posts
        FROM sentiment_results sr
        JOIN raw_documents rd ON sr.raw_document_id = rd.id
        WHERE sr.company_id = ? AND rd.published_at IS NOT NULL
        GROUP BY date(rd.published_at)
        ORDER BY date(rd.published_at)
        """
        ts_df = get_data(time_series_query, params=(selected_company_id,))
        
        if not ts_df.empty:
            ts_df['date'] = pd.to_datetime(ts_df['date'])
            
            fig_ts = px.line(
                ts_df, x='date', y='avg_sentiment', markers=True,
                title=f"Evolución del Sentimiento: {selected_company_str}",
                labels={'date': 'Fecha', 'avg_sentiment': 'Sentimiento Promedio'},
            )
            # Agregar tamaño al scatter para representar volumen
            fig_ts.add_scatter(x=ts_df['date'], y=ts_df['avg_sentiment'], mode='markers', 
                               marker=dict(size=ts_df['posts']*3, color=ts_df['avg_sentiment'], colorscale='RdYlGn'),
                               name='Volumen de Posts')
            
            st.plotly_chart(fig_ts, use_container_width=True)
            
            # Tabla de últimos posts crudos para auditar
            st.subheader("Últimos Posts Analizados")
            recent_posts_query = """
            SELECT rd.published_at as Fecha, sr.sentiment_label as Etiqueta, 
                   sr.sentiment_score as Score, rd.text as Texto
            FROM sentiment_results sr
            JOIN raw_documents rd ON sr.raw_document_id = rd.id
            WHERE sr.company_id = ?
            ORDER BY rd.published_at DESC
            LIMIT 10
            """
            recent_posts_df = get_data(recent_posts_query, params=(selected_company_id,))
            st.dataframe(recent_posts_df, use_container_width=True)
            
        else:
            st.info(f"No hay datos temporales suficientes para {selected_company_str}.")
else:
    st.warning("No hay empresas registradas en la base de datos.")
