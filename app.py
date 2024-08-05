
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression
from datetime import datetime
import datetime as dt
import pandas_datareader as pdr
import quandl
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import requests
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import expected_returns, risk_models, DiscreteAllocation, get_latest_prices
import os
import glob
from PIL import Image

FINNHUB_API_KEY = 'cqo132hr01qo886587u0cqo132hr01qo886587ug'

def translate_text(text, dest_language='en'):
    """Traduire le texte en utilisant Google Translate."""
    try:
        translated = translator.translate(text, dest=dest_language)
        return translated.text
    except Exception as e:
        st.error(f"Erreur lors de la traduction : {e}")
        return text

def get_finnhub_news():
    url = f'https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Erreur lors de la récupération des nouvelles.")
        return []

def display_finnhub_news():
    news = get_finnhub_news()
    if news:
        st.subheader('Actualités Financières')
        # Limiter à 10 articles les plus récents
        top_news = news[:5]
        for article in top_news:
            title = article.get('headline', 'Pas de titre')
            link = article.get('url', '#')
            summary = article.get('summary', 'Résumé non disponible')
            timestamp = article.get('datetime', '')
            formatted_date = datetime.fromtimestamp(timestamp).strftime('%d %b %Y %H:%M:%S') if timestamp else 'Date non disponible'

            st.markdown(f"""
                <div style="margin-bottom: 15px; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px;">
                    <h3 style="margin: 0; font-size: 16px;"><a href="{link}" target="_blank" style="text-decoration: none; color: #1f77b4;">{title}</a></h3>
                    <p style="margin: 5px 0; color: #555;">{summary}</p>
                    <p style="margin: 5px 0; font-size: 12px; color: #888;">{formatted_date}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Aucune nouvelle disponible pour le moment.")


# Initialize session state attributes
if 'available_expirations' not in st.session_state:
    st.session_state.available_expirations = []

# Function to download stock data
def download_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    info = stock.info
    stock_name = info.get('shortName', info.get('longName', ticker))
    return data['Close'], stock_name, info

# Function to display company info
def display_company_info(info):

    st.write(f"**Description**: {info.get('longBusinessSummary', 'N/A')}")
    st.write(f"**Nom**: {info.get('longName', 'N/A')}")
    st.write(f"**Symbole**: {info.get('symbol', 'N/A')}")
    st.write(f"**Nom Court**: {info.get('shortName', 'N/A')}")
    st.write(f"**Secteur**: {info.get('sector', 'N/A')}")
    st.write(f"**Industrie**: {info.get('industry', 'N/A')}")
    st.write(f"**Pays**: {info.get('country', 'N/A')}")
    st.write(f"**Site Web**: {info.get('website', 'N/A')}")

# Fonction pour Simulation de Monte Carlo
def monte_carlo_simulation(data, num_simulations, num_days):
    returns = (data / data.shift(1) - 1).dropna()
    last_price = data[-1]
    simulation_df = pd.DataFrame()

    for x in range(num_simulations):
        daily_vol = returns.std()
        price_series = []
        price = last_price * np.exp((0.5 * daily_vol**2) + daily_vol * norm.ppf(np.random.rand()))
        price_series.append(price)

        for y in range(num_days - 1):
            price = price_series[-1] * np.exp((0.5 * daily_vol**2) + daily_vol * norm.ppf(np.random.rand()))
            price_series.append(price)

        simulation_df[x] = price_series

    return simulation_df, last_price, simulation_df.iloc[-1].mean()

# Function to calculate historical volatility
def calculate_historical_volatility(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    historical_data = stock.history(start=start_date, end=end_date)
    returns = np.log(historical_data['Close'] / historical_data['Close'].shift(1))
    volatility = returns.std() * np.sqrt(252)
    return volatility

# Function to get the risk-free rate
def get_risk_free_rate():
    try:
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=365)
        risk_free_rate_data = pdr.get_data_fred('DGS10', start_date, end_date)
        latest_rate = risk_free_rate_data['DGS10'].iloc[-1] / 100
        return latest_rate
    except Exception as e:
        st.error(f"Erreur lors de la récupération du taux sans risque: {str(e)}")
        return None

# Function to fetch option data
def fetch_option_data(ticker, expiry_date):
    stock = yf.Ticker(ticker)
    available_expirations = stock.options
    if expiry_date and expiry_date not in available_expirations:
        st.error(f"La date d'expiration `{expiry_date}` n'est pas disponible.")
        return None, None, None, None, None, available_expirations

    options = stock.option_chain(expiry_date) if expiry_date else None
    if options:
        calls = options.calls
        S = stock.history(period='1d')['Close'].iloc[-1]
        r = get_risk_free_rate() if get_risk_free_rate() is not None else 0.01
        strikes = calls['strike'].values
        market_prices = calls['lastPrice'].values
        expiration = dt.datetime.strptime(expiry_date, '%Y-%m-%d')
        T = (expiration - dt.datetime.now()).days / 365.0
        return S, strikes, market_prices, T, r, available_expirations
    return None, None, None, None, None, available_expirations

# Function for Black-Scholes call price
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Function for implied volatility
def implied_volatility(S, K, T, r, market_price):
    def objective_function(sigma):
        return black_scholes_call(S, K, T, r, sigma) - market_price

    try:
        iv = brentq(objective_function, 1e-5, 3)
    except ValueError:
        iv = np.nan
    return iv

# Function to predict stock prices
def predict_stock_prices(ticker, forecast_days=7):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='1y')
    hist['Return'] = hist['Close'].pct_change()

    for lag in range(1, forecast_days + 1):
        hist[f'Lag{lag}'] = hist['Return'].shift(lag)
    hist.dropna(inplace=True)

    feature_cols = [f'Lag{lag}' for lag in range(1, forecast_days + 1)]
    X = hist[feature_cols]
    y = hist['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    forecast_features = np.array([hist[feature_cols].iloc[-1]])
    predicted_price = model.predict(forecast_features)

    y_pred_test = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mean_price = y_test.mean()
    win_rate = 1 - (rmse / mean_price)

    return predicted_price, win_rate

# Function to plot prediction
def plot_prediction(ticker, forecast_days, predicted_price, win_rate):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='1y')

    # Calculate 30-day moving average
    hist['30D_MA'] = hist['Close'].rolling(window=30).mean()

    # Define the future date
    future_date = hist.index[-1] + pd.Timedelta(days=forecast_days)

    fig = go.Figure()

    # Add candlestick chart for historical data
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close'],
        name='Prix Historique'
    ))

    # Add 30-day moving average
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['30D_MA'],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Moyenne Mobile 30 Jours'
    ))

    # Add predicted price as a scatter trace
    fig.add_trace(go.Scatter(
        x=[future_date],
        y=predicted_price,
        mode='markers',
        marker=dict(color='red', size=10),
        name=f'Prix Prédit dans {forecast_days} Jours'
    ))

    # Add annotation for predicted price
    fig.add_annotation(
        x=future_date,
        y=predicted_price[0],
        text=f'Prix Prédit: ${predicted_price[0]:.2f}\nTaux de Réussite: {win_rate:.2%}',
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        font=dict(size=12, color='red'),
        align='center'
    )

    fig.update_layout(
        title=f'Prédiction des Prix de {ticker} pour les {forecast_days} Prochains Jours',
        xaxis_title='Date',
        yaxis_title='Prix',
        xaxis_rangeslider_visible=False,
        plot_bgcolor='white',
        hovermode='x unified'
    )

    fig.add_annotation(
        text='guccipepito',
        xref='paper', yref='paper',
        x=0.01, y=0.01,
        showarrow=False,
        font=dict(size=12, color='black'),
        align='left'
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)

# Fonction pour ajouter une régression linéaire
def plot_linear_regression(data):
    # Prepare the data
    data = data.dropna().reset_index()
    data['Date_Ordinal'] = data['Date'].map(pd.Timestamp.toordinal)

    X = data[['Date_Ordinal']]
    y = data['Close']

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)

    # Create the Plotly figure
    fig = go.Figure()

    # Add historical price trace
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Close'],
        mode='lines',
        name='Prix Historique',
        line=dict(color='blue')
    ))

    # Add regression line trace
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=predictions,
        mode='lines',
        name='Régression Linéaire',
        line=dict(color='red', dash='dash')
    ))

    # Update layout
    fig.update_layout(
        title='Régression Linéaire sur les Prix de Clôture',
        xaxis_title='Date',
        yaxis_title='Prix',
        plot_bgcolor='white',
        hovermode='x unified'
    )

    # Add annotation in the bottom left
    fig.add_annotation(
        text='guccipepito',
        xref='paper', yref='paper',
        x=0.01, y=0.01,
        showarrow=False,
        font=dict(
            size=10,
            color='black'
        ),
        align='left',
        opacity=0.5
    )

    # Compute regression metrics
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)

    

    # Show the plot in Streamlit
    st.plotly_chart(fig)

# Fonction utilisant le Machine Learning pour prédire la valeur d'un actif
def predict_stock_prices_advanced(ticker, forecast_days=7):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='1y')

    hist['Return'] = hist['Close'].pct_change()
    for lag in range(1, forecast_days + 1):
        hist[f'Lag{lag}'] = hist['Return'].shift(lag)
    hist.dropna(inplace=True)

    feature_cols = [f'Lag{lag}' for lag in range(1, forecast_days + 1)]
    X = hist[feature_cols]
    y = hist['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    latest_data = pd.DataFrame([X.iloc[-1]], columns=feature_cols)
    prediction = model.predict(latest_data)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    win_rate = 1 - np.sqrt(mse) / y_test.mean()

    print(f"Predicted price for {ticker} in {forecast_days} days: ${prediction[0]:.2f}")
    print(f"Model Win Rate: {win_rate:.2%}")
    return prediction, win_rate

# Fonction pour télécharger l'historique des prix de clôture ajustés
def get_price_history(ticker, sdate, edate):
    data = yf.download(ticker, start=sdate, end=edate)['Adj Close']
    return data

# Fonction pour afficher la performance des actions
def plot_performance(prices_df):
    fig = go.Figure()

    for c in prices_df.columns:
        fig.add_trace(go.Scatter(
            x=prices_df.index,
            y=prices_df[c],
            mode='lines',
            name=c
        ))

    fig.update_layout(
        title='Performance des Actions',
        xaxis_title='Date (Années)',
        yaxis_title='Prix USD (Clôture ajustée)',
        legend=dict(
            x=0,
            y=1,
            traceorder='normal',
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='rgba(0,0,0,0.5)',
            borderwidth=1
        ),
        plot_bgcolor='white',
        hovermode='x unified'
    )

    # Ajouter le texte dans le coin inférieur droit
    fig.add_annotation(
        text='guccipepito',
        xref='paper', yref='paper',
        x=0.99, y=0.01,
        showarrow=False,
        font=dict(
            size=10,
            color='black'
        ),
        align='right',
        opacity=0.5
    )

    # Afficher la grille pour l'axe des ordonnées
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    # Ajouter les outils d'interaction (zoom, dézoom, déplacement)
    fig.update_layout(
        dragmode='pan',  # Permet de déplacer le graphique
        xaxis=dict(
            rangeslider=dict(visible=True),  # Ajouter un curseur pour zoomer sur l'axe des x
            showspikes=True,  # Afficher les pointillés lors du survol
            spikemode='across',  # Les pointillés traversent l'axe
        ),
        yaxis=dict(
            showspikes=True,
            spikemode='across',
        )
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)

# Fonction pour afficher la frontière efficiente
def plot_efficient_frontier(prices_df):
    returns_df = prices_df.pct_change()[1:]

    # Calcul du VaR historique
    confidence_level = 0.95
    VaR = returns_df.quantile(1 - confidence_level)

    # vecteur de rendement et matrice de covariance
    r = ((1 + returns_df).prod()) ** (252 / len(returns_df)) - 1
    cov = returns_df.cov() * 252
    e = np.ones(len(r))

    # calculer les rendements historiques moyens des actifs
    mu = expected_returns.mean_historical_return(prices_df)

    # Calculer la matrice de covariance échantillon des rendements des actifs
    S = risk_models.sample_cov(prices_df)
    S = risk_models.CovarianceShrinkage(prices_df).ledoit_wolf()
    S = (S + S.T) / 2

    # Créer un objet Frontière Efficiente
    ef = EfficientFrontier(mu, S)

    # optimiser pour le ratio de Sharpe maximum
    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    latest_prices = get_latest_prices(prices_df)
    weights = cleaned_weights

    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=3000)
    allocation, leftover = da.greedy_portfolio()

    # Calculer la frontière efficiente
    icov = np.linalg.inv(cov)
    h = np.matmul(e, icov)
    g = np.matmul(r, icov)
    a = np.sum(e * h)
    b = np.sum(r * h)
    c = np.sum(r * g)
    d = a * c - b**2

    # portefeuille de tangence minimum et variance
    mvp = h / a
    mvp_returns = b / a
    mvp_risk = (1 / a) ** (1 / 2)

    # portefeuille de tangence
    tagency = g / b
    tagency_returns = c / b
    tagency_risk = c ** (1 / 2) / b

    min_expected_return = mu.min()
    max_expected_return = mu.max()
    exp_returns = np.linspace(min_expected_return, max_expected_return, num=100)
    risk = ((a * exp_returns ** 2 - 2 * b * exp_returns + c) / d) ** (1 / 2)

    # Tracé de la ligne de marché des titres (SML)
    SML_slope = 1 / c**(1 / 2)
    SML_risk = exp_returns * SML_slope

    # Création du graphique avec Plotly
    fig = go.Figure()

    # Frontière efficiente
    fig.add_trace(go.Scatter(x=risk, y=exp_returns, mode='lines', name='Frontière Efficiente',
                             line=dict(color='blue', dash='dash')))

    # Ligne de marché des titres (SML)
    fig.add_trace(go.Scatter(x=SML_risk, y=exp_returns, mode='lines', name='Ligne de Marché des Titres (SML)',
                             line=dict(color='red', dash='dashdot')))

    # Points des portefeuilles
    fig.add_trace(go.Scatter(x=[mvp_risk], y=[mvp_returns], mode='markers',
                             name='Portefeuille de Volatilité Minimale', marker=dict(color='red', size=10, symbol='star')))
    fig.add_trace(go.Scatter(x=[tagency_risk], y=[tagency_returns], mode='markers',
                             name='Portefeuille Optimal en Risque', marker=dict(color='green', size=10, symbol='star')))

    # Mise en forme du graphique
    fig.update_layout(
        title="Frontière Efficiente & Ligne de Marché des Titres",
        xaxis_title="Écart-type (Risque)",
        yaxis_title="Rendement Attendu",
        legend=dict(
            x=0.99, y=0.01,
            xanchor='right', yanchor='bottom',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.5)',
            font=dict(color='black')
        ),
        plot_bgcolor='white',
        hovermode='x unified'
    )

    # Ajouter une grille pour l'axe des ordonnées
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    # Ajouter le texte "guccipepito" en bas à gauche
    fig.add_annotation(
        text="guccipepito",
        xref="paper", yref="paper",
        x=0.01, y=0.01,
        showarrow=False,
        font=dict(size=10, color="black"),
        opacity=0.5
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)

    # Résolution du problème de rendement cible
    target_return = tagency_returns
    target_risk = tagency_risk

    if target_return < mvp_returns:
        optimal_portfolio = mvp
        optimal_return = mvp_returns
        optimal_risk = mvp_risk
    else:
        l = (c - b * target_return) / d
        m = (a * target_return - b) / d
        optimal_portfolio = l * h + m * g
        optimal_return = np.sum(optimal_portfolio * r)
        optimal_risk = ((a * optimal_return ** 2 - 2 * b * optimal_return + c) / d) ** (1 / 2)

    # Récupération des performances du portefeuille
    annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(verbose=True)

    # Création du graphique pour les performances du portefeuille et poids nettoyés
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Texte des performances du portefeuille
    performance_text = f"Rendement annuel attendu : {annual_return * 100:.1f}%\n" \
                       f"Volatilité annuelle : {annual_volatility * 100:.1f}%\n" \
                       f"Ratio de Sharpe : {sharpe_ratio:.2f}"
    axs[0, 0].text(0.1, 0.5, performance_text, fontsize=12, ha='left', va='center')
    axs[0, 0].axis('off')
    axs[0, 0].set_title('Performances du Portefeuille')

    # Graphique des poids nettoyés du portefeuille
    axs[0, 1].bar(weights.keys(), weights.values())
    axs[0, 1].set_title('Poids Nettoyés du Portefeuille')
    axs[0, 1].set_xlabel('Actifs')
    axs[0, 1].set_ylabel('Poids')
    axs[0, 1].tick_params(axis='x', rotation=45)

    # Graphique de l'allocation discrète des actifs
    axs[1, 0].pie(list(allocation.values()), labels=list(allocation.keys()), autopct='%1.1f%%', startangle=140)
    axs[1, 0].set_title('Allocation Discrète des Actifs')

    # Graphique des fonds restants après l'allocation
    axs[1, 1].text(0.5, 0.5, f"Fonds restants :\n{leftover:.2f} CAD", fontsize=14, ha='center', va='center')
    axs[1, 1].axis('off')
    axs[1, 1].set_title('Fonds Restants')

    # Ajout de la légende spécifique en bas à droite
    fig.text(0.95, 0.05, 'guccipepito', fontsize=12, color='black', ha='right', va='bottom', alpha=0.5)

    # Ajustement de l'espacement entre les sous-graphiques
    plt.tight_layout()

    # Affichage du graphique
    plt.show()

    # Impression de l'allocation discrète du portefeuille et des fonds restants
    print(f"Allocation Discrète du Portefeuille: {allocation}")
    print(f"Fonds Restants: {leftover:.2f}")

# Function to plot volatility surface
def plot_volatility_surface(ticker, expiry_date, forecast_days):
    S, strikes, market_prices, T, r, _ = fetch_option_data(ticker, expiry_date)
    if S is None:
        return

    ivs_list = []
    maturities_list = []

    for day in range(forecast_days + 1):
        T_forecast = T + day / 365.0
        ivs = [implied_volatility(S, K, T_forecast, r, P) for K, P in zip(strikes, market_prices)]
        ivs_list.append(ivs)
        maturities_list.append([T_forecast] * len(strikes))

    ivs_array = np.array(ivs_list)
    maturities_array = np.array(maturities_list)

    if ivs_array.size == 0 or strikes.size == 0:
        st.write("Données insuffisantes pour tracer la surface.")
        return

    strike_grid, maturity_grid = np.meshgrid(strikes, np.linspace(T, T + forecast_days / 365.0, forecast_days + 1))
    fig = go.Figure(data=[go.Surface(z=ivs_array, x=strike_grid, y=maturity_grid, colorscale='Viridis')])
    fig.update_layout(
        title=f'Surface de Volatilité Implicite pour {ticker}',
        scene=dict(
            xaxis_title='Prix d\'Exercice',
            yaxis_title='Échéance (Années)',
            zaxis_title='Volatilité Implicite'
        )
    )
    fig.add_annotation(
        text="guccipepito",
        xref="paper", yref="paper",
        x=0.01, y=0.01,
        showarrow=False,
        font=dict(size=10, color="white"),
        opacity=0.5
    )
    st.plotly_chart(fig)

def download_bond_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    info = stock.info
    bond_name = info.get('shortName', info.get('longName', ticker))
    return data['Close'], bond_name, info

def plot_sinusoidal_with_ticker(ticker, start_date, end_date, amplitude=1.0, period=30):
    """
    Trace les prix de l'action avec une vague sinusoïdale superposée.
    
    Paramètres :
    - ticker : Le symbole de l'action (par ex. 'AAPL').
    - start_date : Date de début des données historiques.
    - end_date : Date de fin des données historiques.
    - amplitude : Amplitude de la vague sinusoïdale.
    - period : Période de la vague sinusoïdale.
    """
    # Télécharger les données historiques du ticker
    data = yf.download(ticker, start=start_date, end=end_date)
    prices = data['Close']
    
    # Calculer les temps en jours
    time = np.arange(len(prices))
    
    # Générer la vague sinusoïdale
    sinusoidal_wave = amplitude * np.sin(2 * np.pi * time / period)
    
    # Créer un graphique Plotly
    fig = go.Figure()
    
    # Tracer les prix de l'action
    fig.add_trace(go.Scatter(
        x=prices.index, 
        y=prices,
        mode='lines',
        name='Prix de l\'action',
        line=dict(color='blue')
    ))
    
    # Tracer la vague sinusoïdale
    fig.add_trace(go.Scatter(
        x=prices.index, 
        y=sinusoidal_wave + np.mean(prices),
        mode='lines',
        name='Vague sinusoïdale',
        line=dict(color='red', dash='dash')
    ))
    
    # Mettre à jour la mise en page du graphique
    fig.update_layout(
        title=f'Vague Sinusoïdale et Prix de l\'Action: {ticker}',
        xaxis_title='Temps',
        yaxis_title='Prix',
        plot_bgcolor='white',
        hovermode='x unified',
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=True),
        yaxis=dict(showgrid=True, zeroline=True),
        width=800,
        height=500
    )
    
    # Afficher le graphique
    fig.show()

def get_stock_prices(tickers):
    prices = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d')  # Dernier jour de données
        if not data.empty:
            prices[ticker] = f"${data['Close'].iloc[-1]:,.2f}"  # Formater le prix
        else:
            prices[ticker] = "N/A"
    return prices

def filter_option_trading_news(news_list):
    """Filtre les nouvelles pour ne garder que celles qui parlent de trading d'options."""
    filtered_news = [article for article in news_list if 'option' in article.get('headline', '').lower()]
    return filtered_news

def display_finnhub_news():
    news = get_finnhub_news()
    if news:
        st.subheader('Actualités Financières')
        # Limiter à 10 articles les plus récents
        top_news = news[:5]
        for article in top_news:
            title = article.get('headline', 'Pas de titre')
            link = article.get('url', '#')
            summary = article.get('summary', 'Résumé non disponible')
            timestamp = article.get('datetime', '')
            formatted_date = datetime.fromtimestamp(timestamp).strftime('%d %b %Y %H:%M:%S') if timestamp else 'Date non disponible'

            st.markdown(f"""
                <div style="margin-bottom: 15px; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px;">
                    <h3 style="margin: 0; font-size: 16px;"><a href="{link}" target="_blank" style="text-decoration: none; color: #1f77b4;">{title}</a></h3>
                    <p style="margin: 5px 0; color: #555;">{summary}</p>
                    <p style="margin: 5px 0; font-size: 12px; color: #888;">{formatted_date}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Aucune nouvelle disponible pour le moment.")

def display_options_news():
    news = get_finnhub_news()
    st.subheader('Top 5 Actualités sur le Trading d\'Options')

    if news:
        option_news = filter_option_trading_news(news)
        if option_news:
            # Limiter à 5 articles
            top_news = option_news[:5]
            for article in top_news:
                title = article.get('headline', 'Pas de titre')
                link = article.get('url', '#')
                summary = article.get('summary', 'Résumé non disponible')
                timestamp = article.get('datetime', '')
                formatted_date = datetime.fromtimestamp(timestamp).strftime('%d %b %Y %H:%M:%S') if timestamp else 'Date non disponible'
                image_url = article.get('image', '')

                st.markdown(f"""
                    <div style="margin-bottom: 20px; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px; background-color: #f9f9f9;">
                        <div style="display: flex; align-items: center;">
                            <img src="{image_url}" alt="Image" style="width: 100px; height: auto; margin-right: 10px; border-radius: 5px;">
                            <div>
                                <h3 style="margin: 0; font-size: 18px;"><a href="{link}" target="_blank" style="text-decoration: none; color: #1f77b4;">{title}</a></h3>
                                <p style="margin: 5px 0; color: #555;">{summary}</p>
                                <p style="margin: 5px 0; font-size: 12px; color: #888;">{formatted_date}</p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Aucune nouvelle sur le trading d'options disponible pour le moment.")
    else:
        st.info("Aucune nouvelle disponible pour le moment.")

def display_economic_news():
    news = get_finnhub_news()
    st.subheader('Top 5 Actualités Économiques')

    if news:
        # Filtrer les nouvelles économiques
        economic_news = [article for article in news if 'economy' in article.get('headline', '').lower()]
        # Limiter à 5 articles
        top_news = economic_news[:5]
        
        if top_news:
            for article in top_news:
                title = article.get('headline', 'Pas de titre')
                link = article.get('url', '#')
                summary = article.get('summary', 'Résumé non disponible')
                timestamp = article.get('datetime', '')
                formatted_date = datetime.fromtimestamp(timestamp).strftime('%d %b %Y %H:%M:%S') if timestamp else 'Date non disponible'
                image_url = article.get('image', '')

                st.markdown(f"""
                    <div style="margin-bottom: 20px; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px; background-color: #f9f9f9;">
                        <div style="display: flex; align-items: center;">
                            <img src="{image_url}" alt="Image" style="width: 100px; height: auto; margin-right: 10px; border-radius: 5px;">
                            <div>
                                <h3 style="margin: 0; font-size: 18px;"><a href="{link}" target="_blank" style="text-decoration: none; color: #1f77b4;">{title}</a></h3>
                                <p style="margin: 5px 0; color: #555;">{summary}</p>
                                <p style="margin: 5px 0; font-size: 12px; color: #888;">{formatted_date}</p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Aucune nouvelle économique disponible pour le moment.")
    else:
        st.info("Aucune nouvelle disponible pour le moment.")

def filter_bond_market_news(news_list):
    """Filtre les nouvelles pour ne garder que celles sur le marché obligataire."""
    filtered_news = [article for article in news_list if 'bond' in article.get('headline', '').lower() or 'obligation' in article.get('headline', '').lower()]
    return filtered_news

def display_bond_market_news():
    news = get_finnhub_news()
    st.subheader('Top 5 Actualités sur le Marché Obligataire')

    if news:
        bond_news = filter_bond_market_news(news)
        # Limiter à 5 articles
        top_news = bond_news[:5]
        
        if top_news:
            for article in top_news:
                title = article.get('headline', 'Pas de titre')
                link = article.get('url', '#')
                summary = article.get('summary', 'Résumé non disponible')
                timestamp = article.get('datetime', '')
                formatted_date = datetime.fromtimestamp(timestamp).strftime('%d %b %Y %H:%M:%S') if timestamp else 'Date non disponible'
                image_url = article.get('image', '')

                st.markdown(f"""
                    <div style="margin-bottom: 20px; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px; background-color: #f9f9f9;">
                        <div style="display: flex; align-items: center;">
                            <img src="{image_url}" alt="Image" style="width: 100px; height: auto; margin-right: 10px; border-radius: 5px;">
                            <div>
                                <h3 style="margin: 0; font-size: 18px;"><a href="{link}" target="_blank" style="text-decoration: none; color: #1f77b4;">{title}</a></h3>
                                <p style="margin: 5px 0; color: #555;">{summary}</p>
                                <p style="margin: 5px 0; font-size: 12px; color: #888;">{formatted_date}</p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Aucune nouvelle sur le marché obligataire disponible pour le moment.")
    else:
        st.info("Aucune nouvelle disponible pour le moment.")

def display_finnhub_news_ticker(ticker):
    news = get_finnhub_news_ticker
    if news:
        st.subheader(f'Actualités pour {ticker}')
        # Limiter à 5 articles les plus récents
        top_news = news[:5]
        for article in top_news:
            title = article.get('headline', 'Pas de titre')
            link = article.get('url', '#')
            summary = article.get('summary', 'Résumé non disponible')
            timestamp = article.get('datetime', '')
            formatted_date = datetime.fromtimestamp(timestamp).strftime('%d %b %Y %H:%M:%S') if timestamp else 'Date non disponible'

            st.markdown(f"""
                <div style="margin-bottom: 15px; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px;">
                    <h3 style="margin: 0; font-size: 16px;"><a href="{link}" target="_blank" style="text-decoration: none; color: #1f77b4;">{title}</a></h3>
                    <p style="margin: 5px 0; color: #555;">{summary}</p>
                    <p style="margin: 5px 0; font-size: 12px; color: #888;">{formatted_date}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Aucune nouvelle disponible pour le moment.")

def get_finnhub_news_ticker(ticker):
    url = f'https://finnhub.io/api/v1/company-news?symbol={ticker}&from={dt.date.today() - dt.timedelta(days=30)}&to={dt.date.today()}&token={FINNHUB_API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Erreur lors de la récupération des nouvelles.")
        return []

def display_excel_file(file_path):
    # Charger le fichier Excel
    df = pd.read_excel(file_path)
    # Afficher le DataFrame
    st.write("### Liste Entreprises Intéressantes")
    st.dataframe(df)
# Streamlit app


st.title('StockGenius')

# Sidebar
st.sidebar.title('Menu')
app_mode = st.sidebar.selectbox('Choisissez une section',
                                ['Accueil','Recherche d\'Actions', 'Screener',
                                  'Simulation Monte Carlo', 'Analyse d\'Options',
                                    'Prévision Économique', 'Marché des Obligations',
                                      'Frontière Efficiente', 'Sources'])

# Tabs content
if app_mode == 'Accueil':
    st.header('Accueil')
    image_url = 'https://y.yarn.co/64247b6c-3850-4b21-b0c6-e807b1e8a591_text.gif'
    st.image(image_url, use_column_width=True)
    display_finnhub_news()
   
    st.markdown("""
<h2>Écoutez Bloomberg TV</h2>
<p>Suivez les dernières actualités financières et économiques en direct sur Bloomberg TV :</p>
<a href="https://www.bloomberg.com/live" target="_blank" style="display: inline-block; padding: 10px 20px; font-size: 16px; color: #fff; background-color: #007bff; border-radius: 5px; text-decoration: none;">Écouter Bloomberg TV</a>
""", unsafe_allow_html=True)
    
    st.markdown("""
<h2>Carte des Marchés Finviz</h2>
<p>Consultez la carte interactive des marchés financiers sur Finviz :</p>
<a href="https://finviz.com/map.ashx?t=sec" target="_blank" style="display: inline-block; padding: 10px 20px; font-size: 16px; color: #fff; background-color: #28a745; border-radius: 5px; text-decoration: none;">Voir la Carte Finviz</a>
""", unsafe_allow_html=True)
    # Ajouter le lien vers le screener Value Investing
    st.markdown("""
<h2>Screener Value Investing</h2>
<p>Explorez des opportunités d'investissement avec le screener de Value Investing :</p>
<a href="https://valueinvesting.io/screener" target="_blank" style="display: inline-block; padding: 10px 20px; font-size: 16px; color: #fff; background-color: #17a2b8; border-radius: 5px; text-decoration: none;">Voir le Screener</a>
""", unsafe_allow_html=True)
    
if app_mode == 'Recherche d\'Actions':
    st.header('Recherche d\'Actions')
    ticker = st.text_input('Entrez le symbole du ticker (par ex. AAPL)', '')
    start_date = st.date_input('Date de début', dt.date(2000, 1, 1))
    end_date = st.date_input('Date de fin', dt.date.today())
    forecast_days = st.number_input("Nombre de jours à prédire", min_value=1, max_value=30, value=7)


    if st.button('Télécharger les données'):
        data, stock_name, info = download_stock_data(ticker, start_date, end_date)
        st.write(f"### {stock_name} ({ticker})")
        display_company_info(info)
        # Plot linear regression
        plot_linear_regression(data.to_frame('Close'))
        predicted_price, win_rate = predict_stock_prices_advanced(ticker, forecast_days)
        
        plot_prediction(ticker, forecast_days, predicted_price, win_rate)
        st.write(f"Prix prédit: ${predicted_price[0]:.2f}")
        st.write(f"Taux de réussite: {win_rate:.2%}")
        st.write("""
## **Prédiction Avancée des Prix des Actions**

Ce graphique utilise un modèle de régression de forêt aléatoire pour prédire les prix futurs des actions. Voici comment cela fonctionne :

1. **Préparation des Données** : La fonction récupère les données historiques des prix de l'action pour la dernière année et calcule les rendements journaliers. Elle crée également des variables de retard (lags) pour les rendements afin de capturer les effets des jours précédents sur le prix de clôture.

2. **Entraînement du Modèle** : Les données sont divisées en ensembles d'entraînement et de test. Un modèle de forêt aléatoire est ensuite formé avec ces données pour prédire les prix de clôture futurs.

3. **Prédiction et Évaluation** : La fonction prédit le prix de l'action pour les jours à venir et calcule un taux de réussite basé sur l'erreur quadratique moyenne (MSE). Le taux de réussite indique la précision du modèle.

4. **Résultats** : Le prix prédit pour les jours à venir ainsi que le taux de réussite du modèle sont affichés.

Utilisez cette fonction pour obtenir des prévisions avancées sur les prix des actions et évaluer la performance du modèle de prédiction.
""")
                  
if app_mode == 'Simulation Monte Carlo':
    st.header('Simulation Monte Carlo')
    ticker = st.text_input('Entrez le symbole du ticker (par ex. AAPL)', '')
    start_date = st.date_input('Date de début', dt.date(2000, 1, 1))
    end_date = st.date_input('Date de fin', dt.date.today())
    num_simulations = st.number_input('Nombre de simulations', value=100, min_value=10, max_value=10000)
    num_days = st.number_input('Nombre de jours de prédiction', value=252, min_value=1, max_value=365)

    if st.button('Lancer la simulation'):
        data, _, _ = download_stock_data(ticker, start_date, end_date)
        simulation_df, last_price, mean_price = monte_carlo_simulation(data, num_simulations, num_days)

        st.subheader('Résultats de la Simulation Monte Carlo')
        st.write(f"Prix actuel: ${last_price:.2f}")
        st.write(f"Prix moyen prédit: ${mean_price:.2f}")

        # Plot results
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(simulation_df)
        ax.axhline(y=last_price, color='r', linestyle='--', label=f'Prix actuel: ${last_price:.2f}')
        ax.axhline(y=mean_price, color='g', linestyle='--', label=f'Prix moyen prédit: ${mean_price:.2f}')
        ax.set_title(f'Simulation Monte Carlo pour {ticker}')
        ax.set_xlabel('Jour')
        ax.set_ylabel('Prix')
        ax.legend()
        plt.figtext(0.01, 0.01, 'guccipepito', fontsize=12, color='gray')
        st.pyplot(fig)

        st.write("### Qu'est-ce qu'une simulation de Monte Carlo ?")
        st.write("""
Les analystes peuvent évaluer les rendements possibles du portefeuille de plusieurs façons. L'approche historique, qui est la plus populaire, tient compte de toutes les possibilités qui se sont déjà produites. Cependant, les investisseurs ne devraient pas s'arrêter là. La méthode Monte Carlo est une méthode stochastique (échantillonnage aléatoire des entrées) pour résoudre un problème statistique, et une simulation est une représentation virtuelle d'un problème. La simulation Monte Carlo combine les deux pour nous donner un outil puissant qui nous permet d'obtenir une distribution (tableau) des résultats pour tout problème statistique avec de nombreuses entrées échantillonnées encore et encore.

**CLÉS À EMPORTER**
- La méthode de Monte Carlo utilise un échantillonnage aléatoire d'informations pour résoudre un problème statistique, tandis qu'une simulation est un moyen de démontrer virtuellement une stratégie.
- Combiné, la simulation de Monte Carlo permet à un utilisateur de trouver une foule de résultats pour un problème statistique avec de nombreux points de données échantillonnés à plusieurs reprises.
- La simulation Monte Carlo peut être utilisée dans la finance d'entreprise, la tarification des options et surtout dans la gestion de portefeuille et la planification des finances personnelles.
- En revanche, la simulation est limitée en ce sens qu'elle ne peut pas tenir compte des marchés baissiers, des récessions ou de tout autre type de crise financière qui pourrait avoir un impact sur les résultats potentiels.

### Simulation Monte Carlo Démystifiée
Les simulations de Monte Carlo peuvent être mieux comprises en pensant à une personne qui lance des dés. Un joueur novice qui joue au craps pour la première fois n'aura aucune idée des chances de lancer un six dans n'importe quelle combinaison (par exemple, quatre et deux, trois et trois, un et cinq). Quelles sont les chances de rouler deux trois, également connus sous le nom de « six durs » ? Lancer les dés plusieurs fois, idéalement plusieurs millions de fois, fournirait une distribution représentative des résultats, ce qui nous dira à quel point il est probable qu'un lancer de six soit un six difficile. Idéalement, nous devrions exécuter ces tests efficacement et rapidement, ce qui est exactement ce qu'offre une simulation de Monte Carlo.

Les prix des actifs ou les valeurs futures des portefeuilles ne dépendent pas des lancers de dés, mais parfois les prix des actifs ressemblent à une marche aléatoire. Le problème avec le fait de regarder l'histoire seule est qu'elle ne représente, en fait, qu'un seul rouleau, ou résultat probable, qui peut ou non être applicable à l'avenir. Une simulation Monte Carlo prend en compte un large éventail de possibilités et nous aide à réduire l'incertitude. Une simulation Monte Carlo est très flexible ; elle nous permet de varier les hypothèses de risque sous tous les paramètres et donc de modéliser une gamme de résultats possibles. On peut comparer plusieurs résultats futurs et personnaliser le modèle à divers actifs et portefeuilles en cours d'examen.

Une simulation Monte Carlo peut tenir compte d'une variété d'hypothèses de risque dans de nombreux scénarios et est donc applicable à toutes sortes d'investissements et de portefeuilles.

### Application de la simulation Monte Carlo
La simulation Monte Carlo a de nombreuses applications dans la finance et d'autres domaines. Monte Carlo est utilisé dans le financement des entreprises pour modéliser les composantes des flux de trésorerie du projet, qui sont affectées par l'incertitude. Le résultat est une fourchette de valeurs actuelles nettes (VNI) ainsi que des observations sur la VAN moyenne de l'investissement analysé et sa volatilité. L'investisseur peut ainsi estimer la probabilité que la VAN soit supérieure à zéro. Monte Carlo est utilisé pour la tarification des options où de nombreux chemins aléatoires pour le prix d'un actif sous-jacent sont générés, chacun ayant un gain associé. Ces gains sont ensuite actualisés jusqu'au présent et moyens pour obtenir le prix de l'option. Il est également utilisé pour la tarification des titres à revenu fixe et des dérivés de taux d'intérêt. Mais la simulation Monte Carlo est utilisée le plus largement dans la gestion de portefeuille et la planification financière personnelle.

### Utilisations dans la gestion de portefeuille
Une simulation Monte Carlo permet à un analyste de déterminer la taille du portefeuille dont un client aurait besoin à la retraite pour soutenir son mode de vie de retraite souhaité et d'autres cadeaux et legs souhaités. Elle tient compte de la distribution des taux de réinvestissement, des taux d'inflation, des rendements de la classe d'actifs, des taux d'imposition et même de la durée de vie possible. Le résultat est une répartition de la taille du portefeuille avec les probabilités de répondre aux besoins de dépenses souhaités du client.

L'analyste utilise ensuite la simulation Monte Carlo pour déterminer la valeur et la distribution attendues d'un portefeuille à la date de départ à la retraite du propriétaire. La simulation permet à l'analyste de prendre une vue multi-périodes et de tenir compte de la dépendance du chemin ; la valeur du portefeuille et la répartition des actifs à chaque période dépendent des rendements et de la volatilité de la période précédente. L'analyste utilise diverses allocations d'actifs avec différents degrés de risque, différentes corrélations entre les actifs et la distribution d'un grand nombre de facteurs - y compris l'épargne à chaque période et la date de retraite - pour arriver à une distribution de portefeuilles ainsi qu'à la probabilité d'arriver à la valeur de portefeuille souhaitée à la retraite. Les différents taux de dépenses et la durée de vie du client peuvent être pris en compte pour déterminer la probabilité que le client soit à court de fonds (la probabilité de risque de ruine ou de longévité) avant son décès.

Le profil de risque et de rendement d'un client est le facteur le plus important qui influence les décisions de gestion de portefeuille. Les rendements requis de la cliente sont fonction de ses objectifs de retraite et de dépenses ; son profil de risque est déterminé par sa capacité et sa volonté de prendre des risques. Le plus souvent, le rendement souhaité et le profil de risque d'un client ne sont pas synchronisés. Par exemple, le niveau de risque acceptable pour un client peut rendre impossible ou très difficile d'obtenir le rendement souhaité. De plus, un montant minimum peut être nécessaire avant la retraite pour atteindre les objectifs du client, mais le mode de vie du client ne permettrait pas l'épargne ou le client peut être réticent à la changer.

### Exemple de simulation Monte Carlo
Considérons un exemple de jeune couple de travailleurs qui travaille très dur et qui a un mode de vie somptueux, y compris des vacances coûteuses chaque année. Ils ont pour objectif de prendre leur retraite de dépenser 170 000 $ par an (environ 14 000 $/mois) et de laisser une succession d'un million de dollars à leurs enfants. Un analyste exécute une simulation et constate que son épargne par période est insuffisante pour construire la valeur de portefeuille souhaitée à la retraite ; cependant, il est réalisable si l'allocation aux actions à petite capitalisation est doublée (jusqu'à 50 à 70 % de 25 à 35 %), ce qui augmentera considérablement leur risque. Aucune des alternatives ci-dessus (économies plus élevées ou risque accru) n'est acceptable pour le client. Ainsi, l'analyste tient compte d'autres ajustements avant d'exécuter à nouveau la simulation. l'analyste retarde sa retraite de deux ans et réduit ses dépenses mensuelles après la retraite à 12 500 $. La distribution qui en résulte montre que la valeur du portefeuille souhaitée est réalisable en augmentant l'allocation aux actions à petite capitalisation de seulement 8 %. Avec les informations disponibles, l'analyste conseille aux clients de retarder la retraite et de diminuer marginalement leurs dépenses, ce que le couple accepte.

### En fin de compte
Une simulation Monte Carlo permet aux analystes et aux conseillers de convertir les chances d'investissement en choix. L'avantage de Monte Carlo est sa capacité à prendre en compte une gamme de valeurs pour divers intrants ; c'est aussi son plus grand inconvénient dans le sens où les hypothèses doivent être justes parce que la production n'est aussi bonne que les intrants. Un autre grand inconvénient est que la simulation de Monte Carlo a tendance à sous-estimer la probabilité d'événements extrêmes comme une crise financière. En fait, les experts soutiennent qu'une simulation comme le Monte Carlo est incapable de tenir compte des aspects comportementaux de la finance et de l'irrationalité manifestée par les acteurs du marché. C'est cependant un outil utile pour les conseillers.
""")

if app_mode == 'Analyse d\'Options':
    st.header('Analyse d\'Options')
    ticker = st.text_input('Entrez le symbole du ticker (par ex. AAPL)', '')
    expiry_date = st.selectbox('Date d\'expiration', st.session_state.available_expirations)

    if st.button('Mettre à jour les dates d\'expiration'):
        _, _, _, _, _, available_expirations = fetch_option_data(ticker, expiry_date)
        st.session_state.available_expirations = available_expirations

    if st.button('Afficher les options'):
        S, strikes, market_prices, T, r, _ = fetch_option_data(ticker, expiry_date)
        if S is not None:
            ivs = [implied_volatility(S, K, T, r, P) for K, P in zip(strikes, market_prices)]
            option_data = pd.DataFrame({
                'Strike': strikes,
                'Prix du marché': market_prices,
                'Volatilité implicite': ivs
            })
            st.write(option_data)
            plot_volatility_surface(ticker, expiry_date, 30)

            # Plot implied volatility vs strike price
            fig_volatility = go.Figure()
            fig_volatility.add_trace(go.Scatter(
                x=option_data['Strike'],
                y=option_data['Volatilité implicite'],
                mode='lines+markers',
                name='Volatilité implicite',
                line=dict(color='blue')
            ))
            fig_volatility.update_layout(
                title='Volatilité Implicite en Fonction du Prix d\'Exercice',
                xaxis_title='Prix d\'exercice',
                yaxis_title='Volatilité implicite',
                plot_bgcolor='white'
            )
            st.plotly_chart(fig_volatility)

            # Plot implied volatility vs market price
            fig_market_price = go.Figure()
            fig_market_price.add_trace(go.Scatter(
                x=option_data['Prix du marché'],
                y=option_data['Volatilité implicite'],
                mode='markers',
                name='Volatilité implicite',
                marker=dict(color='red')
            ))
            fig_market_price.update_layout(
                title='Volatilité Implicite en Fonction du Prix du Marché',
                xaxis_title='Prix du marché',
                yaxis_title='Volatilité implicite',
                plot_bgcolor='white'
            )
            st.plotly_chart(fig_market_price)


            st.write("""
### Surface de Volatilité

La surface de volatilité est un graphique 3D illustrant les volatilités implicites des options d'une action selon les prix d'exercice et les dates d'expiration. Elle révèle les variations de volatilité qui ne sont pas capturées par les modèles simples comme Black-Scholes.

**Points Clés :**
- La surface de volatilité montre comment la volatilité implicite varie avec le prix d'exercice et le temps jusqu'à l'expiration.
- La volatilité implicite, utilisée dans la tarification des options, indique l'attente du marché sur la volatilité future de l'action.
- Le modèle Black-Scholes, malgré ses hypothèses, est souvent incorrect. La surface de volatilité, non uniforme et dynamique, illustre ces écarts.

**Principes des Options :**
- **Option d'appel** : Droit d'acheter l'action à un prix spécifique.
- **Option de vente** : Droit de vendre l'action à un prix spécifique.
- **Options Européennes vs Américaines** : Européennes exécutables à expiration, Américaines à tout moment avant l'expiration.

**Tarification des Options :**
- Le modèle Black-Scholes suppose une absence de dividende, des marchés efficaces, pas de commissions, des taux d'intérêt constants, et des rendements log-normaux.

**Asymétrie de Volatilité :**
- Les options de vente ont souvent des volatilités implicites plus élevées que prévu, en raison des achats de puts pour couverture.

**Volatilité Locale :**
- Représente la volatilité dans une petite zone de la surface globale.

**Structure du Terme :**
- Montre comment la volatilité implicite varie avec les mois d'expiration pour une même grève.

**Conclusion :**
- La surface de volatilité montre les limites du modèle Black-Scholes, mais reste un outil utile dans l'analyse des options.
""")

if app_mode == 'Prévision Économique':
    st.header('Prévision Économique')
    country = st.selectbox('Choisissez un pays', ['États-Unis', 'Canada'])
    api_key = st.text_input('API', 'rBnQyygXVXNxBvqqFBY1')

    if api_key and st.button('Prévoir'):
        data_gdp = None
        data_unemployment = None
        data_inflation = None

        if country == 'États-Unis':
            data_gdp = quandl.get("FRED/GDP", authtoken=api_key)
            data_unemployment = quandl.get("FRED/UNRATE", authtoken=api_key)
            data_inflation = quandl.get("FRED/CPIAUCSL", authtoken=api_key)
        elif country == 'Canada':
            data_gdp = quandl.get("ODA/CAN_NGDPD", authtoken=api_key)
            data_unemployment = quandl.get("ODA/CAN_LUR", authtoken=api_key)
            data_inflation = quandl.get("ODA/CAN_PCPI", authtoken=api_key)

        if data_gdp is not None:
            st.subheader("Produit Intérieur Brut (PIB)")
            
            # Calcul des moyennes mobiles
            data_gdp['30_Day_MA'] = data_gdp['Value'].rolling(window=30).mean()
            data_gdp['100_Day_MA'] = data_gdp['Value'].rolling(window=100).mean()
            
            # Tracer les données avec les moyennes mobiles
            fig_gdp = go.Figure()
            fig_gdp.add_trace(go.Scatter(
                x=data_gdp.index,
                y=data_gdp['Value'],
                mode='lines',
                name='PIB',
                line=dict(color='blue')
            ))
            fig_gdp.add_trace(go.Scatter(
                x=data_gdp.index,
                y=data_gdp['30_Day_MA'],
                mode='lines',
                name='Moyenne Mobile 30 Jours',
                line=dict(color='orange', dash='dash')
            ))
            fig_gdp.add_trace(go.Scatter(
                x=data_gdp.index,
                y=data_gdp['100_Day_MA'],
                mode='lines',
                name='Moyenne Mobile 100 Jours',
                line=dict(color='green', dash='dash')
            ))
            fig_gdp.update_layout(
                title='Produit Intérieur Brut avec Moyennes Mobiles',
                xaxis_title='Date',
                yaxis_title='PIB',
                plot_bgcolor='white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_gdp)
            
            # Analyse des statistiques
            last_value = data_gdp['Value'].iloc[-1]
            max_value = data_gdp['Value'].max()
            min_value = data_gdp['Value'].min()
            
            pct_change = data_gdp['Value'].pct_change()
            growth_mean = pct_change.mean() * 100
            annual_growth = (data_gdp['Value'].iloc[-1] / data_gdp['Value'].iloc[-60] - 1) * 100 if len(data_gdp) > 60 else float('nan')
            volatility = pct_change.std() * 100
            negative_growth_count = (pct_change < 0).sum()
            annual_growth_change = data_gdp['Value'].pct_change(periods=4).mean() * 100
            trend_last_value = data_gdp['Value'].rolling(window=12).mean().iloc[-1]
            
            st.write("**Analyse historique :**")
            st.write(f"- Valeur actuelle : {last_value:,.2f}")
            st.write(f"- Valeur maximale sur la période : {max_value:,.2f}")
            st.write(f"- Valeur minimale sur la période : {min_value:,.2f}")
            st.write(f"- Croissance annuelle moyenne : {growth_mean:.2f}%")
            st.write(f"- Taux de croissance du PIB sur les 5 dernières années : {annual_growth:.2f}%")
            st.write(f"- Variabilité du PIB (écart type) : {volatility:.2f}%")
            st.write(f"- Nombre de périodes de croissance négative : {negative_growth_count}")
            st.write(f"- Taux de croissance du PIB en glissement annuel : {annual_growth_change:.2f}%")
            st.write(f"- Tendances observées : {trend_last_value:,.2f}")
            
            # Exemple simple de prévision avec régression linéaire
            from sklearn.linear_model import LinearRegression
            data_gdp_reset = data_gdp.reset_index()
            data_gdp_reset['Date_Ordinal'] = pd.to_datetime(data_gdp_reset['Date']).map(pd.Timestamp.toordinal)
            X = data_gdp_reset[['Date_Ordinal']]
            y = data_gdp_reset['Value']
            model = LinearRegression()
            model.fit(X, y)
            predicted_value = model.predict([[data_gdp_reset['Date_Ordinal'].iloc[-1]]])[0]
            st.write(f"- Modèle de prévision simple : {predicted_value:,.2f}")

        if data_unemployment is not None:
            st.subheader("Taux de Chômage")
            
            # Calcul des moyennes mobiles
            data_unemployment['30_Day_MA'] = data_unemployment['Value'].rolling(window=30).mean()
            data_unemployment['100_Day_MA'] = data_unemployment['Value'].rolling(window=100).mean()
            
            # Tracer les données avec les moyennes mobiles
            fig_unemployment = go.Figure()
            fig_unemployment.add_trace(go.Scatter(
                x=data_unemployment.index,
                y=data_unemployment['Value'],
                mode='lines',
                name='Taux de Chômage',
                line=dict(color='blue')
            ))
            fig_unemployment.add_trace(go.Scatter(
                x=data_unemployment.index,
                y=data_unemployment['30_Day_MA'],
                mode='lines',
                name='Moyenne Mobile 30 Jours',
                line=dict(color='orange', dash='dash')
            ))
            fig_unemployment.add_trace(go.Scatter(
                x=data_unemployment.index,
                y=data_unemployment['100_Day_MA'],
                mode='lines',
                name='Moyenne Mobile 100 Jours',
                line=dict(color='green', dash='dash')
            ))
            fig_unemployment.update_layout(
                title='Taux de Chômage avec Moyennes Mobiles',
                xaxis_title='Date',
                yaxis_title='Taux de Chômage',
                plot_bgcolor='white',
                hovermode='x unified'
            )
            st.plotly_chart(fig_unemployment)
            
            # Analyse des statistiques
            last_value = data_unemployment['Value'].iloc[-1]
            max_value = data_unemployment['Value'].max()
            min_value = data_unemployment['Value'].min()
            
            pct_change = data_unemployment['Value'].pct_change()
            growth_mean = pct_change.mean() * 100
            annual_growth = (data_unemployment['Value'].iloc[-1] / data_unemployment['Value'].iloc[-60] - 1) * 100 if len(data_unemployment) > 60 else float('nan')
            volatility = pct_change.std() * 100
            negative_growth_count = (pct_change < 0).sum()
            annual_growth_change = data_unemployment['Value'].pct_change(periods=4).mean() * 100
            trend_last_value = data_unemployment['Value'].rolling(window=12).mean().iloc[-1]
            
            st.write("**Analyse historique :**")
            st.write(f"- Taux actuel : {last_value:.2f}%")
            st.write(f"- Taux maximal sur la période : {max_value:.2f}%")
            st.write(f"- Taux minimal sur la période : {min_value:.2f}%")
            st.write(f"- Variation annuelle moyenne : {growth_mean:.2f}%")
            st.write(f"- Nombre de mois avec des augmentations du taux de chômage : {(data_unemployment.pct_change() > 0).sum()}")
            st.write(f"- Nombre de mois avec des baisses du taux de chômage : {(data_unemployment.pct_change() < 0).sum()}")
            st.write(f"- Taux de chômage sur les 5 dernières années : {annual_growth:.2f}%")
            st.write(f"- Écart type du taux de chômage : {volatility:.2f}%")
            
            # Exemple simple de prévision avec régression linéaire
            model = LinearRegression()
            data_unemployment_reset = data_unemployment.reset_index()
            data_unemployment_reset['Date_Ordinal'] = pd.to_datetime(data_unemployment_reset['Date']).map(pd.Timestamp.toordinal)
            X = data_unemployment_reset[['Date_Ordinal']]
            y = data_unemployment_reset['Value']
            model.fit(X, y)
            predicted_value = model.predict([[data_unemployment_reset['Date_Ordinal'].iloc[-1]]])[0]
            st.write(f"- Modèle de prévision simple : {predicted_value:.2f}")

        if data_inflation is not None:
            st.subheader("Inflation")
            
            # Calcul des moyennes mobiles
            data_inflation['30_Day_MA'] = data_inflation['Value'].rolling(window=30).mean()
            data_inflation['100_Day_MA'] = data_inflation['Value'].rolling(window=100).mean()
            
            # Tracer les données avec les moyennes mobiles
            fig_inflation = go.Figure()
            fig_inflation.add_trace(go.Scatter(
                x=data_inflation.index,
                y=data_inflation['Value'],
                mode='lines',
                name='Inflation',
                line=dict(color='blue')
            ))
            fig_inflation.add_trace(go.Scatter(
                x=data_inflation.index,
                y=data_inflation['30_Day_MA'],
                mode='lines',
                name='Moyenne Mobile 30 Jours',
                line=dict(color='orange', dash='dash')
            ))
            fig_inflation.add_trace(go.Scatter(
                x=data_inflation.index,
                y=data_inflation['100_Day_MA'],
                mode='lines',
                name='Moyenne Mobile 100 Jours',
                line=dict(color='green', dash='dash')
            ))
            fig_inflation.update_layout(
                title='Inflation avec Moyennes Mobiles',
                xaxis_title='Date',
                yaxis_title='Inflation',
                plot_bgcolor='white',
                hovermode='x unified'
            )
            st.plotly_chart(fig_inflation)
            
            # Analyse des statistiques
            last_value = data_inflation['Value'].iloc[-1]
            max_value = data_inflation['Value'].max()
            min_value = data_inflation['Value'].min()
            
            pct_change = data_inflation['Value'].pct_change()
            growth_mean = pct_change.mean() * 100
            annual_growth = (data_inflation['Value'].iloc[-1] / data_inflation['Value'].iloc[-60] - 1) * 100 if len(data_inflation) > 60 else float('nan')
            volatility = pct_change.std() * 100
            negative_growth_count = (pct_change < 0).sum()
            annual_growth_change = data_inflation['Value'].pct_change(periods=4).mean() * 100
            trend_last_value = data_inflation['Value'].rolling(window=12).mean().iloc[-1]
            
            st.write("**Analyse historique :**")
            st.write(f"- Valeur actuelle : {last_value:.2f}")
            st.write(f"- Valeur maximale sur la période : {max_value:.2f}")
            st.write(f"- Valeur minimale sur la période : {min_value:.2f}")
            st.write(f"- Croissance annuelle moyenne : {growth_mean:.2f}%")
            st.write(f"- Taux d'inflation sur les 5 dernières années : {annual_growth:.2f}%")
            st.write(f"- Variabilité de l'inflation (écart type) : {volatility:.2f}%")
            st.write(f"- Nombre de mois avec une inflation négative : {negative_growth_count}")
            st.write(f"- Croissance annuelle moyenne de l'inflation : {annual_growth_change:.2f}%")
            st.write(f"- Tendances observées : {trend_last_value:.2f}")
            
            # Exemple simple de prévision avec régression linéaire
            model = LinearRegression()
            data_inflation_reset = data_inflation.reset_index()
            data_inflation_reset['Date_Ordinal'] = pd.to_datetime(data_inflation_reset['Date']).map(pd.Timestamp.toordinal)
            X = data_inflation_reset[['Date_Ordinal']]
            y = data_inflation_reset['Value']
            model.fit(X, y)
            predicted_value = model.predict([[data_inflation_reset['Date_Ordinal'].iloc[-1]]])[0]
            st.write(f"- Modèle de prévision simple : {predicted_value:.2f}")
            display_economic_news()

    else:
        st.error("Impossible de récupérer les données. Vérifiez votre clé API et réessayez.")

if app_mode == 'Marché des Obligations':
    st.header('Marché des Obligations')

    # Télécharger les données d'obligation
    bond_ticker = st.text_input('Entrez le ticker de l\'obligation (par ex. TLT)', 'TLT')
    start_date = st.date_input('Date de début', dt.date(2022, 1, 1))
    end_date = st.date_input('Date de fin', dt.date.today())

    if st.button('Télécharger les données'):
        data, bond_name, info = download_bond_data(bond_ticker, start_date, end_date)
        st.write(f"### {bond_name} ({bond_ticker})")
        st.line_chart(data)

        # Ajouter la régression linéaire
        data_df = data.to_frame(name='Close')
        plot_linear_regression(data_df)

        # Stocker les données dans le session_state pour utilisation future
        st.session_state['bond_data'] = data
        st.session_state['bond_name'] = bond_name
        display_bond_market_news()

    # Comparer avec un indice de référence
    if 'bond_data' in st.session_state:
        comparison_ticker = st.text_input('Entrez le ticker de l\'indice de référence (par ex. US10Y)', 'US10Y')
        if st.button('Comparer avec l\'Indice de Référence'):
            comparison_data, comparison_name, _ = download_bond_data(comparison_ticker, start_date, end_date)
            st.write(f"### Comparaison avec {comparison_name} ({comparison_ticker})")

            fig_comparison = go.Figure()
            fig_comparison.add_trace(go.Scatter(
                x=st.session_state['bond_data'].index,
                y=st.session_state['bond_data'],
                mode='lines',
                name=f'{st.session_state["bond_name"]} ({bond_ticker})',
                line=dict(color='blue')
            ))
            fig_comparison.add_trace(go.Scatter(
                x=comparison_data.index,
                y=comparison_data,
                mode='lines',
                name=f'{comparison_name} ({comparison_ticker})',
                line=dict(color='red')
            ))
            fig_comparison.update_layout(
                title='Comparaison avec l\'Indice de Référence',
                xaxis_title='Date',
                yaxis_title='Prix de Clôture',
                plot_bgcolor='white',
                hovermode='x unified'
            )
            st.plotly_chart(fig_comparison)
            
        
if app_mode == 'Frontière Efficiente':
    # Entrée de tickers sous forme de chaîne de caractères
    st.header('Frontière Efficiente')
    tickers_input = st.text_input("Entrez les tickers (séparés par des virgules)", "BFH, SDE.TO, AAPL")
    ticker = [ticker.strip() for ticker in tickers_input.split(',')]  # Convertir en liste de tickers
    start = st.date_input('Date de début', dt.date(2022, 1, 1))
    end = st.date_input('Date de fin', dt.date.today())
    prices_df = get_price_history(' '.join(ticker), sdate=start, edate=end)
    returns_df = prices_df.pct_change()[1:]
    st.subheader("Performance des Actions")
    plot_performance(prices_df)

    st.subheader("Frontière Efficiente")
    plot_efficient_frontier(prices_df)

if app_mode == "Screener":
    st.title("Screener")
    file_path = 'Copie de export-6.xlsx'
    display_excel_file(file_path)

if app_mode == "Sources":
    st.title("Sources")

    st.write("""
    ### Importance des Sources de Qualité
    Avoir des sources de qualité est crucial pour obtenir des informations fiables et précises, particulièrement dans le domaine de l'investissement. Les sources de qualité fournissent des données vérifiées et des analyses approfondies, ce qui aide à prendre des décisions éclairées et à éviter les pièges des informations erronées ou biaisées.
    """)

    st.write("Voici quelques sources de qualité pour vos recherches :")

    st.write("[Investopedia](https://www.investopedia.com)")
    st.write("[Banque centrale américaine (Federal Reserve)](https://www.federalreserve.gov)")
    st.write("[Banque du Canada (Bank of Canada)](https://www.bankofcanada.ca)")
    st.write("[Questrade](https://www.questrade.com)")
    st.write("[Seeking Alpha](https://seekingalpha.com)")
    st.write("[Zacks](https://www.zacks.com)")
    st.write("[Sedar](https://www.sedarplus.ca)")