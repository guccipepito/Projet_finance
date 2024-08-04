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
from pypfopt import expected_returns, risk_models, EfficientFrontier, DiscreteAllocation, get_latest_prices
import os
import glob
from PIL import Image


# Initialize session state attributes
if 'available_expirations' not in st.session_state:
    st.session_state.available_expirations = []

def display_company_info(info):
    st.write("### Informations sur l'Entreprise")
    st.write(f"**Nom**: {info.get('longName', 'N/A')}")
    st.write(f"**Symbole**: {info.get('symbol', 'N/A')}")
    st.write(f"**Nom Court**: {info.get('shortName', 'N/A')}")
    st.write(f"**Secteur**: {info.get('sector', 'N/A')}")
    st.write(f"**Industrie**: {info.get('industry', 'N/A')}")
    st.write(f"**Description**: {info.get('longBusinessSummary', 'N/A')}")
    st.write(f"**Pays**: {info.get('country', 'N/A')}")
    st.write(f"**Site Web**: {info.get('website', 'N/A')}")


# Function to download stock data
def download_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    info = stock.info
    stock_name = info.get('shortName', info.get('longName', ticker))
    return data['Close'], stock_name, info

# Function to display company info
def display_company_info(info):
    st.write("### Informations sur l'Entreprise")
    st.write(f"**Nom**: {info.get('longName', 'N/A')}")
    st.write(f"**Symbole**: {info.get('symbol', 'N/A')}")
    st.write(f"**Nom Court**: {info.get('shortName', 'N/A')}")
    st.write(f"**Secteur**: {info.get('sector', 'N/A')}")
    st.write(f"**Industrie**: {info.get('industry', 'N/A')}")
    st.write(f"**Description**: {info.get('longBusinessSummary', 'N/A')}")
    st.write(f"**Pays**: {info.get('country', 'N/A')}")
    st.write(f"**Site Web**: {info.get('website', 'N/A')}")
    
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
    dates = hist.index

    future_date = dates[-1] + pd.Timedelta(days=forecast_days)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(dates, hist['Close'], label='Prix Historique', color='blue')
    ax.plot(future_date, predicted_price, 'ro', label=f'Prix Prédit dans {forecast_days} Jours')

    ax.annotate(f'Prix Prédit: ${predicted_price[0]:.2f}\nTaux de Réussite: {win_rate:.2%}',
                xy=(future_date, predicted_price), xytext=(future_date, predicted_price[0] + 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, color='red', ha='center')

    ax.set_title(f'Prédiction des Prix de {ticker} pour les {forecast_days} Prochains Jours', fontsize=16, ha='center')
    ax.set_xlabel('Date', fontsize=14, ha='center')
    ax.set_ylabel('Prix', fontsize=14, ha='center')
    ax.legend()
    ax.grid(True)

    plt.figtext(0.01, 0.01, f'Note: Les prédictions sont basées sur un modèle d\'apprentissage automatique et peuvent varier.', wrap=True, horizontalalignment='left', fontsize=12)
    st.pyplot(fig)

def plot_linear_regression(data):
    data = data.dropna().reset_index()
    data['Date_Ordinal'] = data['Date'].map(pd.Timestamp.toordinal)
    
    X = data[['Date_Ordinal']]
    y = data['Close']
    
    model = LinearRegression()
    model.fit(X, y)
    
    predictions = model.predict(X)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data['Date'], data['Close'], label='Prix Historique', color='blue')
    ax.plot(data['Date'], predictions, label='Régression Linéaire', color='red', linestyle='--')
    
    ax.set_title('Régression Linéaire sur les Prix de Clôture', fontsize=16, ha='center')
    ax.set_xlabel('Date', fontsize=14, ha='center')
    ax.set_ylabel('Prix', fontsize=14, ha='center')
    ax.legend()
    ax.grid(True)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)
    
    st.write(f"**Pente**: {slope:.2f}")
    st.write(f"**Intercept**: {intercept:.2f}")
    st.write(f"**R² (Coefficient de Détermination)**: {r_squared:.2f}")
    
    return fig

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
    plt.figure(figsize=(10, 6))
    for c in prices_df.columns:
        plt.plot(prices_df.index, prices_df[c], label=c)

    plt.title('Performance des Actions')
    plt.xlabel('Date (Années)', fontsize=10)
    plt.ylabel('Prix USD (Clôture ajustée)', fontsize=10)
    plt.legend(prices_df.columns.values, loc='upper left')
    plt.grid(axis='y')
    plt.text(0.99, 0.01, 'guccipepito', transform=plt.gca().transAxes,
             fontsize=10, color='black', ha='right', va='bottom', alpha=0.5)
    plt.tight_layout()
    st.pyplot(plt)

# Fonction pour calculer la frontière efficiente et tracer les graphiques
def plot_efficient_frontier(prices_df):
    returns_df = prices_df.pct_change()[1:]

    # Calcul du VaR historique
    confidence_level = 0.95  # Niveau de confiance pour le VaR
    VaR = returns_df.quantile(1 - confidence_level)

    # vecteur de rendement et matrice de covariance
    r = ((1 + returns_df).prod()) ** (252 / len(returns_df)) - 1  # calcul des rendements annuels
    cov = returns_df.cov() * 252  # matrice de covariance annualisée
    e = np.ones(len(r))  # vecteur de uns de la même longueur que le nombre d'actions

    # calculer les rendements historiques moyens des actifs
    mu = expected_returns.mean_historical_return(prices_df)  # rendement historique moyen

    # Calculer la matrice de covariance échantillon des rendements des actifs
    S = risk_models.sample_cov(prices_df)
    S = risk_models.CovarianceShrinkage(prices_df).ledoit_wolf()
    S = (S + S.T) / 2

    # Créer un objet Frontière Efficiente en utilisant les rendements moyens et la matrice de covariance
    ef = EfficientFrontier(mu, S)

    # trouver les poids du portefeuille qui maximisent le ratio de Sharpe
    raw_weights = ef.max_sharpe()  # optimiser pour le ratio de Sharpe maximum
    cleaned_weights = ef.clean_weights()  # nettoyer les poids bruts
    latest_prices = get_latest_prices(prices_df)  # obtenir les derniers prix des actifs
    weights = cleaned_weights  # assigner les poids nettoyés au portefeuille

    # créer un objet Allocation Discrète avec les poids du portefeuille, les derniers prix et la valeur totale du portefeuille
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

    # Tracé de l'efficience frontière
    plt.figure(figsize=(10, 6))
    plt.plot(risk, exp_returns, linestyle='--', color='b', linewidth=2, label='Frontière Efficient')
    plt.scatter(mvp_risk, mvp_returns, marker='*', color='r', s=200, label='Portefeuille de Volatilité Minimale')
    plt.scatter(tagency_risk, tagency_returns, marker='*', color='g', s=200, label='Portefeuille Optimal en Risque')
    plt.title("Frontière Efficiente", fontsize=14)
    plt.xlabel("Écart-type (Risque)", fontsize=12)
    plt.ylabel("Rendement Attendu", fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc="lower right", fontsize=10)
    plt.text(0.01, 0.01, 'guccipepito', transform=plt.gca().transAxes,
             fontsize=10, color='black', ha='left', va='bottom', alpha=0.5)
    plt.tight_layout()
    st.pyplot(plt)

    # Titre et étiquettes des axes avec des tailles de police adaptées
    plt.title("Frontière Efficiente & Ligne de Marché des Titres", fontsize=14)
    plt.xlabel("Écart-type (Risque)", fontsize=12)
    plt.ylabel("Rendement Attendu", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Quadrillage sur l'axe y

    # Légende avec position optimisée dans le coin inférieur droit
    plt.legend(["Frontière Efficiente", "Ligne de Marché des Titres (SML)", "Portefeuille de Volatilité Minimale", "Portefeuille Optimal en Risque"], loc="lower right", fontsize=10)

    # Ajout de "guccipepito" en bas à gauche en noir
    plt.text(0.01, 0.01, 'guccipepito', transform=plt.gca().transAxes,
        fontsize=10, color='black', ha='left', va='bottom', alpha=0.5)

    # Ajustement automatique de la mise en page
    plt.tight_layout()
    st.pyplot(plt)

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
    st.plotly_chart(fig)

# Streamlit app
st.title('Plateforme d\'Analyse Financière')

# Sidebar
st.sidebar.title('Menu')
app_mode = st.sidebar.selectbox('Choisissez une section', 
                                ['Accueil','Recherche d\'Actions', 'Simulation Monte Carlo', 'Analyse d\'Options', 'Prévision Économique', 'Marché des Obligations', 'Frontière Efficiente'])

# Tabs content

if app_mode == 'Accueil':
    st.header('Accueil')
    # Description de l'application
    st.write("""
    ## Bienvenue dans l'application pour l'investisseur intelligent

    Cette application est conçue pour les investisseurs qui cherchent à prendre des décisions éclairées en utilisant des outils d'analyse avancés. Inspirée des principes de l'investissement intelligent, elle offre des fonctionnalités pour évaluer, analyser et optimiser les portefeuilles d'investissement.

    ### Référence

    - Graham, Benjamin. *The Intelligent Investor*. HarperCollins, 2003.

    ### Citations Inspirantes

    - **Warren Buffett** : "Le risque vient de ne pas savoir ce que vous faites." - Warren Buffett

    - **Michael Burry** : "Il est très difficile de prévoir la direction du marché, mais vous pouvez prévoir la direction des affaires de l'entreprise." - Michael Burry

    - **Jim Simons** : "Les marchés sont très irrationnels à court terme, mais sur le long terme, ils se comportent de manière beaucoup plus prévisible." - Jim Simons
    """)
    

    
    
    

if app_mode == 'Recherche d\'Actions':
    st.header('Recherche d\'Actions')
    ticker = st.text_input('Entrez le symbole du ticker (par ex. AAPL)', 'AAPL')
    start_date = st.date_input('Date de début', dt.date(2022, 1, 1))
    end_date = st.date_input('Date de fin', dt.date.today())
    forecast_days = st.number_input("Nombre de jours à prédire", min_value=1, max_value=30, value=7)
   

    if st.button('Télécharger les données'):
        data, stock_name, info = download_stock_data(ticker, start_date, end_date)
        st.write(f"### {stock_name} ({ticker})")
        display_company_info(info)
        predicted_price, win_rate = predict_stock_prices_advanced(ticker, forecast_days)
        st.write(f"### Prédiction du prix de l'action pour {forecast_days} jours")
        st.write(f"Prix prédit: ${predicted_price[0]:.2f}")
        st.write(f"Taux de réussite: {win_rate:.2%}")
        plot_prediction(ticker, forecast_days, predicted_price, win_rate)
        st.write(f"### {stock_name} Historique de clôture {forecast_days} jours")
        st.line_chart(data)
        
        st.write(f"### {stock_name} Régression Linéaire {forecast_days} jours")
        # Plot linear regression
        fig = plot_linear_regression(data.to_frame('Close'))
        st.pyplot(fig)

        
        
        

if app_mode == 'Simulation Monte Carlo':
    st.header('Simulation Monte Carlo')
    ticker = st.text_input('Entrez le symbole du ticker (par ex. AAPL)', 'AAPL')
    start_date = st.date_input('Date de début', dt.date(2022, 1, 1))
    end_date = st.date_input('Date de fin', dt.date.today())
    num_simulations = st.number_input('Nombre de simulations', value=100, min_value=10, max_value=1000)
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
    ticker = st.text_input('Entrez le symbole du ticker (par ex. AAPL)', 'AAPL')
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

            fig, ax = plt.subplots()
            ax.plot(option_data['Strike'], option_data['Volatilité implicite'], 'bo-', label='Volatilité implicite')
            ax.set_xlabel('Prix d\'exercice')
            ax.set_ylabel('Volatilité implicite')
            ax.set_title('Volatilité implicite en fonction du prix d\'exercice')
            ax.legend()
            st.pyplot(fig)

        
            st.write("""
### La surface de volatilité

La surface de volatilité est un graphique en trois dimensions montrant les volatilités implicites des options d'une action qui y sont cotées sur différents prix d'exercice et expirations.

Toutes les options sur la même action n'ont pas la même volatilité implicite (IV). Ces différences existent en raison de divergences dans la façon dont le marché valorise les options d'achat d'actions avec des caractéristiques différentes et ce que les modèles de prix des options d'achat d'actions disent que les prix corrects devraient être.

Pour mieux comprendre ce phénomène, il est important de connaître les bases des options d'achat d'actions, le prix des options d'achat d'actions et la surface de la volatilité.

**CLÉS À EMPORTER**
- La surface de volatilité fait référence à un graphique en trois dimensions des volatilités implicites des différentes options cotées sur la même action.
- La volatilité implicite est utilisée dans la tarification des options pour montrer la volatilité attendue de l'action sous-jacente de l'option sur la durée de vie de l'option.
- Le modèle Black-Scholes est un modèle de tarification des options bien connu qui utilise la volatilité comme l'une des variables de sa formule pour les options de prix.
- La surface de volatilité varie au fil du temps et est loin d'être plate, ce qui démontre que les hypothèses du modèle Black-Scholes ne sont pas toujours correctes.

**Principes de base sur les options d'achat d'actions**
- **Option d'appel** : Une option d'achat donne au propriétaire le droit d'acheter l'action sous-jacente de l'option à un prix prédéterminé spécifique, connu sous le nom de prix d'exercice, au plus tard à une date spécifique, connue sous le nom de date d'expiration. Le propriétaire d'une option de vente réalise un bénéfice lorsque le prix de l'action sous-jacente augmente.
- **Option de vente** : Une option de vente donne au propriétaire le droit de vendre l'action sous-jacente de l'option à un prix spécifique à une date précise ou au plus tard. Le propriétaire d'une option de vente réalise un profit lorsque le prix de l'action sous-jacente diminue.
- **Autres types d'options** : Une option européenne ne peut être exécutée qu'à la date d'expiration, tandis qu'une option américaine peut être exécutée à tout moment jusqu'à la date d'expiration. Il existe également des options des Bermudes, qui sont exécutables à certaines dates prédéfinies.

**Principes de base de la tarification des options**
Le modèle Black-Scholes nécessite six hypothèses pour fonctionner :
1. L'action sous-jacente ne verse pas de dividende et ne le fera jamais.
2. L'option doit être de style européen.
3. Les marchés financiers sont efficaces.
4. Aucune commission n'est facturée sur le commerce.
5. Les taux d'intérêt restent constants.
6. Les rendements des actions sous-jacentes sont distribués de manière log-normale.

La surface de volatilité est un graphique tridimensionnel où l'axe des x est le temps jusqu'à la maturité, l'axe z est le prix d'exercice et l'axe des y est la volatilité implicite. Si le modèle Black-Scholes était tout à fait correct, alors la surface de volatilité à travers les prix d'exercice et le délai d'échéance devrait être stable. En pratique, ce n'est pas le cas.

La surface de volatilité est loin d'être plate et varie souvent au fil du temps parce que les hypothèses du modèle Black-Scholes ne sont pas toujours vraies. Par exemple, les options dont les prix d'exercice sont plus bas ont tendance à avoir des volatilités implicites plus élevées que celles dont les prix d'exercice sont plus élevés.

**Exemple de surface de volatilité**
Pourquoi l'asymétrie de volatilité existe-t-elle ? Depuis la fin des années 1980, les traders d'options ont reconnu que les options de vente à la baisse ont des volatilités implicites sur le marché plus élevées que leurs modèles ne le prévoiraient autrement. Cela est dû au fait que les investisseurs et les commerçants qui sont naturellement longs achèteront des puts de protection à des fins d'assurance, augmentant les prix des ventes par rapport aux options à la hausse.

**Qu'est-ce que la volatilité locale ?**
La volatilité locale considère la volatilité implicite de seulement une petite zone de la surface de volatilité globale. La surface de volatilité peut être considérée comme une agrégation de toutes les volatilités locales dans une chaîne d'options.

**Qu'est-ce que la structure du terme de volatilité ?**
La structure à terme de volatilité fait partie de la surface de volatilité qui décrit comment les options sur la même action présentent des volatilités implicites différentes sur différents mois d'expiration, même pour la même grève. Une structure à terme en pente ascendante indique que les traders s'attendent à ce que l'action sous-jacente devienne plus volatile au fil du temps; une pente descendante indique qu'elle deviendra moins volatile.

**L'essentiel**
Le fait que la surface de volatilité existe montre que le modèle Black-Scholes est loin d'être précis. Cependant, la plupart des sociétés d'investissement et de négoce utilisent toujours le modèle Black-Scholes ou une variante de celui-ci.
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
            st.line_chart(data_gdp)
            st.write(data_gdp.describe())

        if data_unemployment is not None:
            st.subheader("Taux de Chômage")
            st.line_chart(data_unemployment)
            st.write(data_unemployment.describe())

        if data_inflation is not None:
            st.subheader("Indice des Prix à la Consommation (Inflation)")
            st.line_chart(data_inflation)
            st.write(data_inflation.describe())

        # Analyse textuelle
        st.subheader("Analyse")
        st.write("Le PIB des {} montre une tendance...".format(country))
        st.write("Le taux de chômage des {} a fluctué...".format(country))
        st.write("L'inflation des {} indique...".format(country))

    else:
        st.error("Impossible de récupérer les données. Vérifiez votre clé API et réessayez.")

if app_mode == 'Marché des Obligations':
   
    
    st.header('Marché des Obligations')
    bond_ticker = st.text_input('Entrez le ticker de l\'obligation (par ex. TLT)', 'TLT')
    start_date = st.date_input('Date de début', dt.date(2022, 1, 1))
    end_date = st.date_input('Date de fin', dt.date.today())

    if st.button('Télécharger les données'):
        data, bond_name, info = download_stock_data(bond_ticker, start_date, end_date)
        st.write(f"### {bond_name} ({bond_ticker})")
        st.line_chart(data)

        # Plot linear regression
        fig = plot_linear_regression(data.to_frame('Close'))
        st.pyplot(fig)

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
