import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
from io import BytesIO

st.set_page_config(page_title="Monte Carlo Simulator Pro", layout="wide")

st.title("🎯 Monte Carlo Simulator Pro")

# =================== CABEÇALHO E UPLOADER =====================
with st.expander("📂 Carregue ou troque o arquivo de análise"):
    csv_file = st.file_uploader(
        "Selecione o arquivo CSV",
        type=["csv"],
        label_visibility="collapsed"
    )
    if csv_file:
        try:
            # Tenta ler e processar o arquivo
            df = pd.read_csv(csv_file, sep=";")
            
            for col in df.columns:
                if df[col].dtype == "object":
                    try:
                        df[col] = df[col].str.replace(",", ".").astype(float)
                    except (ValueError, AttributeError):
                        pass # Ignora colunas que não podem ser convertidas
            
            # Armazena o DataFrame limpo na memória da sessão
            st.session_state['df'] = df
            
            st.success("✅ Arquivo carregado e processado com sucesso!")

        except Exception as e:
            st.error(f"Ocorreu um erro ao processar o arquivo: {e}")

    # ========== Seleção do método ===========
    metodos = [col for col in df.columns if "P/L" in col]
    metodo = st.selectbox("🎯 Selecione o método para análise:", metodos)
    retornos = df[metodo].dropna().values

    # ========== Configurações de simulação ===========
    col1, col2, col3 = st.columns(3)
    with col1:
        n_simulacoes = st.number_input("🔢 Número de Simulações", value=10000, step=1000)
    with col2:
        stake_pct = st.number_input("💰 Stake (% da banca)", value=1.0, step=0.1) / 100
    with col3:
        banca_inicial = st.number_input("🏦 Banca Inicial", value=1000, step=100)

if 'df' in st.session_state:
    df = st.session_state['df']
    # ========== Filtros Dinâmicos ===========
    filtros = []
    colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

    with st.expander("🛠️ Clique para adicionar filtros"):
        n_filtros = st.number_input("Quantos filtros deseja aplicar?", min_value=0, max_value=10, value=0, step=1)
        for i in range(n_filtros):
            col_f, op_f, val_f = st.columns([4, 2, 4])
            with col_f:
                coluna = st.selectbox(f"Coluna {i+1}", colunas_numericas, key=f"coluna_{i}")
            with op_f:
                operador = st.selectbox(f"Operador {i+1}", ["<=", ">=", "=="], key=f"operador_{i}")
            with val_f:
                valor = st.number_input(f"Valor {i+1}", key=f"valor_{i}")
            filtros.append((coluna, operador, valor))

    # Aplicar filtros
    df_filtrado = df.copy()
    for coluna, operador, valor in filtros:
        if operador == "<=":
            df_filtrado = df_filtrado[df_filtrado[coluna] <= valor]
        elif operador == ">=":
            df_filtrado = df_filtrado[df_filtrado[coluna] >= valor]
        elif operador == "==":
            df_filtrado = df_filtrado[df_filtrado[coluna] == valor]

    retornos_filtrados = df_filtrado[metodo].dropna().values
    lucro_historico = retornos_filtrados.sum()
    saldos_cumulativos = retornos_filtrados.cumsum()
    saldos_cumulativos_com_inicio = np.insert(saldos_cumulativos, 0, 0)
    pico_anterior = np.maximum.accumulate(saldos_cumulativos_com_inicio)
    drawdowns_historicos = pico_anterior - saldos_cumulativos_com_inicio
    drawdown_max_historico = drawdowns_historicos.max()
    roi_historico = (lucro_historico / len(retornos_filtrados)) * 100
    n_apostas = len(retornos_filtrados)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Nº de Apostas", value=n_apostas)
    with col2:
        st.metric(label="Lucro Histórico", value=f"{lucro_historico:.2f} stakes")
    with col3:
        st.metric(label="ROI Histórico", value=f"{roi_historico:.2f}%")
    with col4:
        st.metric(label="Drawdown Máximo", value=f"-{drawdown_max_historico:.2f} stakes")

    st.area_chart(saldos_cumulativos_com_inicio)

    if n_apostas == 0:
        st.warning("❌ Nenhuma aposta encontrada com os filtros definidos.")
    else:
        # ========== Rodar Simulação ==========
        if st.button("🚀 Rodar Simulação Monte Carlo"):
            simulacoes = []
            drawdowns = []

            for _ in range(n_simulacoes):
                amostra = np.random.choice(retornos_filtrados, size=n_apostas, replace=True)
                lucro_acumulado = np.cumsum(amostra)
                simulacoes.append(lucro_acumulado)
                dd = lucro_acumulado - np.maximum.accumulate(lucro_acumulado)
                drawdowns.append(dd.min())

            simulacoes = np.array(simulacoes)
            drawdowns = np.array(drawdowns)
            lucros_finais = simulacoes[:, -1]
            num_simulacoes = len(lucros_finais)
            prob_lucro = (lucros_finais > 0).sum() / num_simulacoes
            volatilidade = lucros_finais.std()

            if lucros_finais.std() > 0:
                sharpe_ratio = lucros_finais.mean() / lucros_finais.std()
            else:
                sharpe_ratio = float('inf')

            # ========== Estatísticas ==========
            st.subheader("📈 Estatísticas")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Lucro Máximo:** {lucros_finais.max():.2f} stakes")
                st.write(f"**Lucro Médio:** {lucros_finais.mean():.2f} stakes")
                st.write(f"**Lucro Mínimo:** {lucros_finais.min():.2f} stakes")
            with col2:
                st.write(f"**Menor Drawdown:** {drawdowns.max():.2f} stakes")
                st.write(f"**Drawdown Médio:** {drawdowns.mean():.2f} stakes")
                st.write(f"**Maior Drawdown:** {drawdowns.min():.2f} stakes")
            with col3:
                st.write(f"**Probabilidade de Lucro (PoP)** {(prob_lucro * 100):.1f}%")
                st.write(f"**Volatilidade (Desvio Padrão):** {volatilidade:.2f}")
                st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")

            # ========== Risco de Ruína ==========
            st.subheader("💣 Risco de Ruína")
            stake_valor = banca_inicial * stake_pct
            ruinas = 0
            for _ in range(1000):
                banca_sim = banca_inicial
                for retorno in np.random.choice(retornos_filtrados, size=n_apostas, replace=True):
                    banca_sim += retorno * stake_valor
                    if banca_sim <= 0:
                        ruinas += 1
                        break
            risco_ruina = (ruinas / 1000) * 100
            st.write(f"**Risco de Ruína (stake {stake_pct*100:.1f}% de {banca_inicial}): {risco_ruina:.2f}%**")

            # ========== Gráficos ==========
            st.subheader("📊 Gráficos de Evolução")

            fig1, ax1 = plt.subplots(figsize=(10, 5))
            for serie in simulacoes:
                ax1.plot(serie, alpha=0.01)
            ax1.axhline(0, color='black', linestyle='--')
            ax1.set_title(f"Monte Carlo - Evolução do Lucro ({n_simulacoes} Simulações)")
            st.pyplot(fig1)

            piores_idx = np.argsort(lucros_finais)[:50]
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            for serie in simulacoes[piores_idx]:
                ax2.plot(serie, alpha=0.2)
            ax2.axhline(0, color='black', linestyle='--')
            ax2.set_title("Monte Carlo - Evolução do Lucro (50 Piores Simulações)")
            st.pyplot(fig2)

            fig3, ax3 = plt.subplots(figsize=(8, 4))
            ax3.hist(lucros_finais, bins=40, edgecolor='black')
            ax3.set_title(f"Distribuição do Lucro Final ({n_simulacoes} Simulações)")
            st.pyplot(fig3)

            fig4, ax4 = plt.subplots(figsize=(8, 4))
            ax4.hist(drawdowns, bins=40, edgecolor='black')
            ax4.set_title(f"Distribuição do Pior Drawdown ({n_simulacoes} Simulações)")
            st.pyplot(fig4)

            # ========== Download dos gráficos ==========
            st.subheader("⬇️ Baixar Gráficos")
            for fig, name in zip([fig1, fig2, fig3, fig4],
                                 ["evolucao_todas_simulacoes.png", "evolucao_50_piores.png",
                                  "histograma_lucros_finais.png", "histograma_drawdowns.png"]):
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=300)
                st.download_button(
                    label=f"📥 Baixar {name}",
                    data=buf.getvalue(),
                    file_name=name,
                    mime="image/png"
                )
