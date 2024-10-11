import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Dashboard de Taxação", layout="wide")
st.title("Dashboard de Análise do Aumento de Taxação ao decorrer do tempo")

# Dados fictícios para análise
@st.cache_data
def load_data():
    np.random.seed(42)
    anos = list(range(2000, 2026))
    regioes = ["Norte", "Nordeste", "Sul", "Sudeste", "Centro-Oeste"]
    
    # Criando um DataFrame com taxas fictícias para cada região ao longo dos anos
    data = []
    for regiao in regioes:
        taxas = np.cumsum(np.random.randint(1, 5, size=len(anos))) + np.random.randint(5, 15)
        data.extend([[ano, regiao, taxa] for ano, taxa in zip(anos, taxas)])
    
    df = pd.DataFrame(data, columns=["Ano", "Região", "Taxa"])
    return df

df = load_data()

# Seção de Filtros no Sidebar
st.sidebar.header("Filtros Personalizados")
anos_selecionados = st.sidebar.slider("Selecionar Intervalo de Anos", min_value=int(df["Ano"].min()), max_value=int(df["Ano"].max()), value=(2000, 2025))
regioes_selecionadas = st.sidebar.multiselect("Selecionar Regiões", options=df["Região"].unique(), default=df["Região"].unique())

# Filtrando dados conforme a seleção
df_filtrado = df[(df["Ano"] >= anos_selecionados[0]) & (df["Ano"] <= anos_selecionados[1])]
df_filtrado = df_filtrado[df_filtrado["Região"].isin(regioes_selecionadas)]

# Colunas para layout
col1, col2 = st.columns([2, 3])

# Tabela de Dados
col1.subheader("Dados Filtrados por Região e Ano")
col1.write("Dados Filtrados, divididos por **região** e **ano**. Assim você pode ajustar os filtros à esquerda para explorar diferentes intervalos de tempo e regiões do país.")
col1.write(df_filtrado)

# Gráfico de Linhas - Tendência de Taxação
col2.subheader("Tendência de Aumento de Taxação por Região")
col2.write("Este gráfico de linhas mostra a **tendência** das taxas ao longo dos anos para cada **região**. Cada ponto no gráfico representa a taxa média de um determinado ano, e as linhas conectam esses pontos para mostrar como as taxas têm aumentado ou diminuído ao longo do tempo.")
fig_linhas = px.line(df_filtrado, x="Ano", y="Taxa", color="Região", markers=True,
                     title="Tendência de Aumento de Taxação por Região")
col2.plotly_chart(fig_linhas, use_container_width=True)

# Gráficos de Comparação entre Anos e Regiões
st.header("Análise Comparativa de Regiões e Anos")

# Gráfico de Barras Empilhadas
st.write("O gráfico abaixo exibe uma **comparação empilhada** das taxas de diferentes regiões ao longo dos anos.")
fig_barras_empilhadas = px.bar(df_filtrado, x="Ano", y="Taxa", color="Região", title="Comparação de Taxação Empilhada por Região")
st.plotly_chart(fig_barras_empilhadas, use_container_width=True)

# Tabela Dinâmica para Visualização de Estatísticas
st.header("Resumo Estatístico Detalhado")
st.write("""A tabela abaixo apresenta um resumo estatístico detalhado para as taxas de cada região.
         
 Alguns termos especificações para melhor entendimento:

- **Número de Dados**: Quantidade de anos considerados para a região.
- **Média**: Taxa média calculada para a região.
- **Variação (Desvio Padrão)**: Mede o quanto os valores variam em torno da média. Quanto maior, mais dispersas estão as taxas.
- **Valores Mínimo e Máximo**: As menores e maiores taxas registradas para cada região.
- **Quartis**: Dividem os dados em quatro partes iguais. O **Primeiro Quartil (25%)** é o valor abaixo do qual 25% dos dados estão, enquanto o **Terceiro Quartil (75%)** indica que 75% dos dados estão abaixo desse valor.
""")
df_estatisticas = df_filtrado.groupby("Região")["Taxa"].describe().reset_index()
df_estatisticas = df_estatisticas.rename(columns={
    "count": "Número de Dados",
    "mean": "Média",
    "std": "Variação",
    "min": "Valor Mínimo",
    "25%": "Primeiro Quartil (25%)",
    "50%": "Mediana (50%)",
    "75%": "Terceiro Quartil (75%)",
    "max": "Valor Máximo"
})
st.write(df_estatisticas)

# Previsão de Taxação usando ARIMA
st.header("Previsão de Taxação para os Próximos Anos")
st.write("""
Abaixo, apresentamos uma **previsão** das taxas para os próximos 5 anos, baseada no modelo **ARIMA**. O ARIMA é um modelo estatístico usado para **análise de séries temporais**, que se ajusta bem a dados históricos e faz previsões com base nas tendências passadas.
""")

# Agrupando dados por ano para análise preditiva
df_ano = df_filtrado.groupby("Ano")["Taxa"].mean().reset_index()

# Treinando o modelo ARIMA
model = ARIMA(df_ano["Taxa"], order=(2, 1, 2))  # Configuração de exemplo
model_fit = model.fit()
forecast = model_fit.forecast(steps=5)  # Previsão para os próximos 5 anos
forecast_anos = list(range(2026, 2031))

# DataFrame com previsões
df_forecast = pd.DataFrame({"Ano": forecast_anos, "Taxa Prevista": forecast})
df_completa = pd.concat([df_ano, df_forecast], ignore_index=True)

# Gráfico com dados históricos e previsão
fig_previsao = go.Figure()
fig_previsao.add_trace(go.Scatter(x=df_ano["Ano"], y=df_ano["Taxa"], mode='lines+markers', name='Taxa Histórica'))
fig_previsao.add_trace(go.Scatter(x=forecast_anos, y=forecast, mode='lines+markers', name='Previsão de Taxa'))
fig_previsao.update_layout(title="Previsão de Taxação para os Próximos Anos", xaxis_title="Ano", yaxis_title="Taxa")
st.plotly_chart(fig_previsao, use_container_width=True)

# Adicionando um Mapa Interativo para Comparar Regiões
st.header("Visualização Geográfica das Taxas por Região")
st.write("""
Utilize o mapa interativo abaixo para visualizar as taxas médias por região. Os pontos no mapa representam as diferentes regiões do Brasil, e o tamanho e a cor dos círculos indicam a **magnitude** da taxa. O mapa facilita uma visão geográfica da distribuição das taxas e como cada região é afetada.
""")

# Dados fictícios para mapeamento de regiões
geo_data = {'Norte': [2.154007, -60.760670], 'Nordeste': [-7.998716, -40.239469],
            'Centro-Oeste': [-15.7801, -47.9292], 'Sudeste': [-22.9068, -43.1729],
            'Sul': [-29.6116, -51.1743]}

df_geo = pd.DataFrame(geo_data).T.reset_index()
df_geo.columns = ["Região", "Lat", "Lon"]
df_geo = df_geo.merge(df_filtrado.groupby("Região").mean().reset_index(), on="Região")

# Mapa Interativo
fig_mapa = px.scatter_mapbox(df_geo, lat="Lat", lon="Lon", size="Taxa", color="Taxa",
                             hover_name="Região", hover_data={"Lat": False, "Lon": False},
                             size_max=15, zoom=3, mapbox_style="carto-positron",
                             title="Mapa Interativo de Taxação por Região")
st.plotly_chart(fig_mapa, use_container_width=True)

# Conclusão e Insights
st.header("Conclusões")
st.write("""
Este dashboard oferece uma visão detalhada sobre o comportamento das taxas ao longo dos anos, separadas por região, analisando acima você poderá criar uma ídeia do que se pode vim ao longo do tempo.""")