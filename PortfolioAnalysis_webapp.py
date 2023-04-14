import streamlit as st
from streamlit_option_menu import option_menu
import yfinance as yf
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from PIL import Image
from datetime import date
# Risk parity library
from numpy.linalg import inv,pinv
from scipy.optimize import minimize
# Excel
import openpyxl
from zipfile import ZipFile

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.sidebar.header("Dashboard")
st.sidebar.markdown('''
---
Created by [Luca Luigi Alberici](https://www.linkedin.com/in/luca-luigi-alberici-37820a21b/) and [Marco Migliardi](https://www.linkedin.com/in/marco-migliardi-b0aa71249/)
''')

with st.sidebar:
    selected = option_menu(
        menu_title= None, # or None if you don't want it
        options=['Portfolio Analysis','Single Stock','Asset Management'],
        icons=['diagram-2-fill','calculator','cash-coin']
    )

if selected == 'Portfolio Analysis':
    st.write(""" # Portfolio Analysis""")
    image = Image.open('fotocopertina.jpg')
    st.image(image, caption='Investments')

if selected == 'Single Stock':
    st.write('Prova - CAMBIARE')
elif selected == 'Asset Management':
    start = st.sidebar.date_input("Starting Date",date.today()+pd.DateOffset(years=-10))
    end = st.sidebar.date_input("Ending Date",date.today())

    if 'dummy_data' not in st.session_state.keys():
        dummy_data = ['AMZN','WMT','MCD','KO','AAPL', 'MSFT', 'GS', 'JPM']
        st.session_state['dummy_data'] = dummy_data
    else:
        dummy_data = st.session_state['dummy_data']

    def checkbox_container(data):
        new_data = st.sidebar.text_input('Enter another ticker to include in your Portfolio')
        cols = st.sidebar.columns(4)
        if cols[0].button('Add'):
            dummy_data.append(new_data)
        if cols[1].button('Remove'):
            dummy_data.remove(new_data)
        if cols[2].button('Select All'):
            for i in data:
                st.session_state['dynamic_checkbox_' + i] = True
            st.experimental_rerun()
        if cols[3].button('UnSelect All'):
            for i in data:
                st.session_state['dynamic_checkbox_' + i] = False
            st.experimental_rerun()
        for i in data:
            st.sidebar.checkbox(i, key='dynamic_checkbox_' + i)

    def get_selected_checkboxes():
        return [i.replace('dynamic_checkbox_','') for i in st.session_state.keys() if i.startswith('dynamic_checkbox_') and st.session_state[i]]

    def rand_weights(n):
        k = np.random.rand(n)
        return k / sum(k)

    listTickers = get_selected_checkboxes()
    listTickers = sorted(listTickers)
    pippo = listTickers.sort()
    checkbox_container(dummy_data)

    startinfo, logret, sintesi, strategy, riskContr, frontier, download = st.tabs(["StartInfo", "LogRet", "Main Variables", "Strategies", "Risk Contribution", "Frontier", "Download"])

    with startinfo:
        if listTickers == list():
            st.write("You have to select at least two tickers")
            df = yf.download("AAPL", start=start, end=end)[
                'Adj Close']
        else:
            dfPrice = yf.download(listTickers, start=start, end=end)[
                'Adj Close']  # estrazione degli Adj.Close dei titoli selezionati

            nasset = len(listTickers)
            st.write(""" ### Interest Rate level """)
            rf = st.slider('Which is the actual interest rate?', 1, 7, 1)
            st.write(""" ### AdjClose """)
            st.write(dfPrice)

    with logret:
        if listTickers == list():
            st.write("You have to select at least one ticker")
        else:
            logretTickers = np.log(dfPrice).diff().dropna()
            st.write(""" ### LogRet """)
            st.write(logretTickers)
    with sintesi:
        if listTickers == list() or len(listTickers) < 2:
            st.write("You have to select at least two tickers")
        else:
            df_expret = logretTickers.mean()
            expret = df_expret.to_numpy()
            expret = np.reshape(expret, (len(expret), 1))

            #st.write(""" # Variances """)
            variances = logretTickers.var()
            stds = np.sqrt(variances)
            #variances

            variances_list = variances.tolist()
            expret_list = df_expret.to_numpy()
            expret_list = expret_list.tolist()
            # expret_list = expret.tolist()
            devstd_list = logretTickers.std()
            devstd_list = devstd_list.to_numpy()
            devstd_list = devstd_list.tolist()
            # #st.write()
            #
            st.write(" ### Expected Return, Variance, StdDev")
            dfMainStat_list = pd.DataFrame({"ExpRet": expret_list,
                                             "Variances": variances_list,
                                             "StdDev": devstd_list})
            dfMainStat_list.index = listTickers
            st.write(dfMainStat_list)

            st.write(""" ### Covariance Matrix""")
            covariance_matrix = logretTickers.cov()
            covariance_matrix  # Matrice di covarianza

            st.write(""" ### Correlation""")
            correlation = logretTickers.corr()
            correlation  # Matrice di correlazione

            #df_variances = pd.DataFrame({listTickers: variances})
            #st.write(type(variances))

            st.write(" ### Heatmatrix")
            fig = plt.figure()
            sns.heatmap(logretTickers.corr(), annot=True, cmap='Reds', center=1, linewidths=.5)
            sns.set(rc={'figure.figsize': (20, 10)})
            fig


    # with frontier:
    #     if listTickers == list() or len(listTickers) < 2:
    #         st.write("You have to select at least two tickers")
    #     else:
    #         st.write(""" ### Frontier """)
            expretMatr = np.asmatrix(expret)
            SigmaMatr = np.asmatrix(covariance_matrix)
            stdsMatr = np.asmatrix(stds)

            def portfolio_performance(w, mu, Sigma):
                mean_prtf = w.T * mu
                std_prtf = np.sqrt(w.T * Sigma * w)
                return std_prtf, mean_prtf

            def generate_random_portfolios(n_prtf, mu, Sigma, rf):
                results = np.zeros((3, n_prtf))
                weight_array = []
                for i in range(n_prtf):
                    weights = rand_weights(nasset).reshape(nasset, 1)
                    weights = np.asmatrix(weights)
                    weight_array.append(weights)

                    portfolio_std_dev, portfolio_return = portfolio_performance(weights, mu, Sigma)

                    results[0, i] = portfolio_std_dev
                    results[1, i] = portfolio_return
                    results[2, i] = (portfolio_return - rf) / portfolio_std_dev
                return results, weight_array

            rf = rf / 25000

            risultati, pesi = generate_random_portfolios(50000, expretMatr, SigmaMatr, rf)

            # Vettore Ones
            ones_vect = np.ones(nasset)
            ones_vect = np.reshape(ones_vect, (len(ones_vect), 1))
            # Matrice Sigma inversa
            icov = np.linalg.inv(SigmaMatr)

            # Parametri per frontiera efficiente
            A = np.matmul(np.matmul(ones_vect.T, icov), expretMatr)
            B = np.matmul(np.matmul(expretMatr.T, icov), expretMatr)
            C = np.matmul(np.matmul(ones_vect.T, icov), ones_vect)
            D = B * C - (A ** 2)

            # Values for plot:
            minRet = np.min(expret) - 0.0001
            maxRet = np.max(expret) + 0.0001

            # FRONTIERA EFFICIENTE
            mup = np.linspace(minRet, maxRet, 1000)
            mup = np.reshape(mup, (1, len(mup)))
            sigmap = (1 / D) * (C * (mup ** 2) - 2 * A * mup + B)
            sigmap = np.sqrt(sigmap)
            vettoremedie = np.array(mup).ravel()
            vettoredevstd = np.array(sigmap).ravel()

            # MINIMUM VARIANCE PORTFOLIO
            mu_mvp = A / C
            mu_mvp = np.array(mu_mvp).ravel()
            sigma_mvp = np.sqrt(1 / C)
            sigma_mvp = np.array(sigma_mvp).ravel()

            # MINIMUM VARIANCE PORTFOLIO
            w_mvp = (1/C)*np.matmul(ones_vect.T, icov)

            # PORTAFOGLI SIMULATI
            sim_sg = risultati[0, :]
            sim_mean = risultati[1, :]
            sim_sr = risultati[2, :]

            # CAPITAL MARKET LINE
            sr_mkt = np.sqrt(B - 2 * A * rf + C * (rf ** 2))
            expretSR = (A / C) - ((D / (C ** 2)) / (rf - (A / C)))
            stdSR = (expretSR - rf) / sr_mkt
            sigmap_cml = np.linspace(0, np.max(stds) + 0.005, 1000)
            mup_cml = rf + sr_mkt * (sigmap_cml)
            vettoremedie_cml = np.array(mup_cml).ravel()
            vettoredevstd_cml = np.array(sigmap_cml).ravel()

            # Per plottare, estraiamo i valori
            expretSR_plot = np.array(expretSR).ravel()
            stdSR_plot = np.array(stdSR).ravel()
            expret_assets = np.array(expretMatr).ravel()
            stds_assets = np.array(stds).ravel()

    with strategy:
        if listTickers == list() or len(listTickers) < 2:
            st.write("You have to select at least two tickers")
        else:
            st.write(""" ### Strategies """)
            er_minus_rf = expret - rf
            z = np.matmul(er_minus_rf.T, icov)
            sum_z = np.sum(z)
            weights_SR = z/sum_z

            check_mean_SR = np.matmul(weights_SR, expretMatr)

            check_std_SR = np.matmul(np.matmul(weights_SR, covariance_matrix), weights_SR.T)
            check_std_SR = np.sqrt(check_std_SR)

            # RISK PARITY PORTFOLIO
            TOLERANCE = 1e-16
            def _allocation_risk(weights, covariances):
                # We calculate the risk of the weights distribution
                portfolio_risk = np.sqrt((weights * covariances * weights.T))[0, 0]
                # It returns the risk of the weights distribution
                return portfolio_risk

            def _assets_risk_contribution_to_allocation_risk(weights, covariances):
                # We calculate the risk of the weights distribution
                portfolio_risk = _allocation_risk(weights, covariances)
                # We calculate the contribution of each asset to the risk of the weights distribution
                assets_risk_contribution = np.multiply(weights.T, covariances * weights.T) \
                                           / portfolio_risk
                # It returns the contribution of each asset to the risk of the weights distribution
                return assets_risk_contribution

            def _risk_budget_objective_error(weights, args):
                # The covariance matrix occupies the first position in the variable
                covariances = args[0]
                # The desired contribution of each asset to the portfolio risk occupies the second position
                assets_risk_budget = args[1]
                # We convert the weights to a matrix
                weights = np.matrix(weights)
                # We calculate the risk of the weights distribution
                portfolio_risk = _allocation_risk(weights, covariances)
                # We calculate the contribution of each asset to the risk of the weights distribution
                assets_risk_contribution = \
                    _assets_risk_contribution_to_allocation_risk(weights, covariances)
                # We calculate the desired contribution of each asset to the risk of the weights distribution
                assets_risk_target = \
                    np.asmatrix(np.multiply(portfolio_risk, assets_risk_budget))
                # Error between the desired contribution and the calculated contribution of each asset
                error = \
                    sum(np.square(assets_risk_contribution - assets_risk_target.T))[0, 0]
                # It returns the calculated error
                return error

            def _get_risk_parity_weights(covariances, assets_risk_budget, initial_weights):
                # Restrictions to consider in the optimisation: only long positions whose sum equals 100%
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                               {'type': 'ineq', 'fun': lambda x: x})
                # Optimisation process in scipy
                optimize_result = minimize(fun=_risk_budget_objective_error,
                                           x0=initial_weights,
                                           args=[covariances, assets_risk_budget],
                                           method='SLSQP',
                                           constraints=constraints,
                                           tol=TOLERANCE,
                                           options={'disp': False})
                # Recover the weights from the optimised object
                weights = optimize_result.x
                # It returns the optimised weights
                return weights

            V = np.matrix(SigmaMatr)
            x_t = [1/nasset]*nasset  # your risk budget percent of total portfolio risk (equal risk)
            w0 = [1/nasset]*nasset

            w_rb = _get_risk_parity_weights(V,x_t,w0)
            w_rb = np.matrix(w_rb)

            w_bar_MVP = list()
            for i in range(nasset):
                w_bar_MVP.append(w_mvp[0,i])

            w_bar_SR = list()
            for i in range(nasset):
                w_bar_SR.append(weights_SR[0,i])

            w_bar_RP = list()
            for i in range(nasset):
                w_bar_RP.append(w_rb[0,i])

            # EQUALLY WEIGHTED
            w_ew = [1 / nasset] * nasset
            w_ew = np.matrix(w_ew)

            w_bar_EW = list()
            for i in range(nasset):
                w_bar_EW.append(w_ew[0,i])

            X_axis = np.arange(nasset)
            figWeights, pltWeights = plt.subplots()
            pltWeights.set_facecolor('white')
            pltWeights.bar(X_axis - 0.2, w_bar_MVP, 0.2, label='MVP')
            pltWeights.bar(listTickers, w_bar_SR, 0.2, label='SR')
            pltWeights.bar(X_axis +0.2, w_bar_RP, 0.2, label='RP')
            pltWeights.bar(X_axis + 0.4, w_bar_EW, 0.2, label='EW')
            pltWeights.legend(handletextpad=0, loc='upper left', prop={'size': 20})
            pltWeights.figure.savefig('Weights.png')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

            # Create a dataFrame using dictionary
            dfWeights = pd.DataFrame({"MVP": w_bar_MVP,
                               "SR": w_bar_SR,
                               "RP": w_bar_RP,
                               "EW": w_bar_EW})
            dfWeights.index = listTickers
            st.write(dfWeights)

            riskContr_RP = _assets_risk_contribution_to_allocation_risk(w_rb, V)
            riskContr_EW = _assets_risk_contribution_to_allocation_risk(w_ew, V)
            riskContr_MVP = _assets_risk_contribution_to_allocation_risk(w_mvp, V)
            riskContr_SR = _assets_risk_contribution_to_allocation_risk(weights_SR, V)

            sigmaRP = _allocation_risk(w_rb, V)
            sigmaEW = _allocation_risk(w_ew, V)
            sigmaMVP = _allocation_risk(w_mvp, V)
            sigmaSR = _allocation_risk(weights_SR, V)

            riskContr_RP = riskContr_RP/sigmaRP
            riskContr_EW = riskContr_EW/sigmaEW
            riskContr_MVP = riskContr_MVP/sigmaMVP
            riskContr_SR = riskContr_SR/sigmaSR

            riskContr_bar_EW = list()
            for i in range(nasset):
                riskContr_bar_EW.append(riskContr_EW[i, 0])

            riskContr_bar_RP = list()
            for i in range(nasset):
                riskContr_bar_RP.append(riskContr_RP[i, 0])

            riskContr_bar_MVP = list()
            for i in range(nasset):
                riskContr_bar_MVP.append(riskContr_MVP[i, 0])

            riskContr_bar_SR = list()
            for i in range(nasset):
                riskContr_bar_SR.append(riskContr_SR[i, 0])

            mu_ew = np.matmul(w_ew, expret)
            mu_rp = np.matmul(w_rb, expret)

            expret_port = [mu_ew[0, 0], mu_mvp[0], mu_rp[0, 0], expretSR[0, 0]]
            Vol_port = [sigmaEW, sigmaMVP, sigmaRP, sigmaSR]

            df_ExpRet_Vol = pd.DataFrame({"MVP": [mu_mvp[0], sigmaMVP],
                                          "SR": [expretSR[0, 0], sigmaSR],
                                          "RP": [mu_rp[0, 0], sigmaRP],
                                          "EW": [mu_ew[0, 0], sigmaEW]}, index=['Expected Return', 'Volatility'])
            st.write(df_ExpRet_Vol, use_container_width=True)

    with riskContr:
        if listTickers == list() or len(listTickers) < 2:
            st.write("You have to select at least two tickers")
        else:
            st.write(""" ### Risk Contributions """)
            index = listTickers
            df_riskContr = pd.DataFrame({'MVP': riskContr_bar_MVP,
                               'SR': riskContr_bar_SR,
                               'RP': riskContr_bar_RP,
                               'EW': riskContr_bar_EW}, index=index)
            axRiskContr = df_riskContr.plot.bar(rot=0)
            axRiskContr.legend(handletextpad=0, loc='upper left', prop={'size': 20})
            axRiskContr.set_facecolor('white')
            axRiskContr.set_xlabel('Assets')
            axRiskContr.set_ylabel('RiskContribution per Strategy')
            axRiskContr.set_title('RiskContribution')
            axRiskContr.figure.savefig('RiskContribution.png')
            st.pyplot()
            st.write(df_riskContr, use_container_width=True)


    with frontier:
        if listTickers == list() or len(listTickers) < 2:
            st.write("You have to select at least two tickers")
        else:
            st.write(""" ### Frontier """)
        # Plot
            figFrontier, axFrontier = plt.subplots()
            axFrontier.set_facecolor('white')
            # Plot simulated portfolios
            axFrontier.scatter(sim_sg, sim_mean, color='grey', label='SimPRTF')
            # Plot Efficient Frontier
            axFrontier.scatter(vettoredevstd, vettoremedie, label='EffFrontier', color='orange')
            # Plot Assets
            axFrontier.scatter(stds_assets, expret_assets, marker='o', color='red', s=300, label='Assets')
            # Plot MVP
            axFrontier.scatter(sigma_mvp, mu_mvp, marker='o', color='orange', s=300, label='MVP')
            # Plot CML
            axFrontier.scatter(vettoredevstd_cml, vettoremedie_cml, color='green', label='CML')
            # Plot SR portfolio
            axFrontier.scatter(stdSR_plot, expretSR_plot, marker='o', color='green', s=300, label='MKT')
            # Plot RP portfolio
            axFrontier.scatter(sigmaRP, mu_rp[0, 0], marker='o', color='blue', s=300, label='RP')
            # Plot EW portfolio
            axFrontier.scatter(sigmaEW, mu_ew[0, 0], marker='o', color='violet', s=300, label='EW')
            # Testo Assets
            for i in range(nasset):
                axFrontier.text(stds_assets[i] + 0.0002, expret_assets[i], listTickers[i], fontsize=20, color='black')
            # ax.text(stds_assets, expret_assets, listTickers, fontsize=20, color='black')
            # Testo MVP
            axFrontier.text(sigma_mvp - 0.001, mu_mvp + 0.00005, "MVP", fontsize=20, color='black')
            # Testo MKT portfolio
            axFrontier.text(stdSR_plot - 0.001, expretSR_plot + 0.00005, "MKT", fontsize=20, color='black')
            # Testo RP portfolio
            axFrontier.text(sigmaRP - 0.001, mu_rp[0, 0] + 0.00003, "RP", fontsize=20, color='black')
            # Testo EW portfolio
            axFrontier.text(sigmaEW - 0.001, mu_ew[0, 0] + 0.00003, "EW", fontsize=20, color='black')
            # Assi:
            axFrontier.set_xlabel('Standard Deviation', fontsize=20, color='black')
            axFrontier.set_ylabel('Expected Return', fontsize=20, color='black')
            axFrontier.set_title('Efficient Frontier', fontweight="bold", size=20)
            # Legend:
            axFrontier.legend(handletextpad=0, loc='upper left', prop={'size': 20})  # Plot legend
            # Plot show:
            axFrontier.figure.savefig('Frontier.png')
            st.pyplot(figFrontier)

    with download:
        if listTickers == list() or len(listTickers) < 2:
            st.write("You have to select at least two tickers")
        else:
            st.write('''You can download:''')
            st.write('''• PortfolioAnalysis.xlsx: it contains prices, logreturns, covariance matrix, correlation matrix, expected
            return, variances, weights, risk contributions ''')
            st.write('''• Figures: frontier, weights, risk contributions ''')

            with pd.ExcelWriter('PortfolioAnalysis.xlsx') as writer:
                dfPrice.to_excel(writer, sheet_name='Prices')
                logretTickers.to_excel(writer, sheet_name='LogRet')
                covariance_matrix.to_excel(writer, sheet_name='CovMatr')
                correlation.to_excel(writer, sheet_name='Corr')
                df_expret.to_excel(writer, sheet_name='ExpRet')
                variances.to_excel(writer, sheet_name='Variances')
                dfWeights.to_excel(writer, sheet_name='Weights')
                df_riskContr.to_excel(writer, sheet_name='RiskContrs')

            # Aprire il file Excel
            workbook = openpyxl.load_workbook('PortfolioAnalysis.xlsx')
            worksheet = workbook['ExpRet']
            worksheet['B1'] = 'ExpRet'
            worksheet = workbook['Variances']
            worksheet['B1'] = 'Variances'
            workbook.save('PortfolioAnalysis.xlsx')

            # Creazione della cartella zip
            with ZipFile('PortfolioAnalysis.zip', 'w') as zip:
                zip.write('PortfolioAnalysis.xlsx')
                zip.write('Frontier.png')
                zip.write('Weights.png')
                zip.write('RiskContribution.png')
            # Download della cartella zip
            with open('PortfolioAnalysis.zip', 'rb') as f:
                bytes = f.read()
                st.download_button(label='Download', data=bytes, file_name='PortfolioAnalysis.zip',
                                   mime='application/zip')
            zip.close()