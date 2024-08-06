# Application d'Analyse et de Prévision des Marchés Financiers

## Description

Cette application interactive permet aux utilisateurs d'analyser les marchés financiers et de réaliser des prévisions éclairées grâce à une gamme d'outils et de fonctionnalités. Le projet est conçu pour les investisseurs intelligents qui souhaitent explorer diverses méthodes d'analyse et de prédiction.

## Fonctionnalités

### Accueil
- **Actualités Financières** : Suivez les dernières nouvelles financières grâce à l'intégration de l'API Finnhub.
- **Bloomberg TV** : Regardez Bloomberg TV en direct pour rester informé des actualités financières importantes.
- **Carte des Marchés Finviz** : Visualisez une carte interactive des marchés financiers via un lien vers Finviz.

### Analyse de Titres
- **Entrée des Données** : Saisissez le symbole boursier, les dates de début et de fin, le nombre de jours à prédire et la période d'analyse.
- **Téléchargement des Données** : Téléchargez les données boursières pour le symbole spécifié et affichez des informations sur l'entreprise.
- **Régression Linéaire** : Tracez la régression linéaire des prix des actions.
- **Prévision avec Apprentissage Automatique** : Prévoyez les prix futurs en utilisant des modèles d'apprentissage automatique et affichez les résultats.
- **Résumé Financier** : Affichez un résumé financier pour le symbole spécifié et calculez le changement de prix sur la période sélectionnée.
- **Sentiments des Investisseurs** : Affichez une jauge des sentiments des investisseurs basée sur le changement de prix.

### Simulation de Monte Carlo
- **Entrée des Données** : Saisissez le symbole boursier, les dates de début et de fin, le nombre de simulations et le nombre de jours de prédiction.
- **Simulation** : Effectuez une simulation de Monte Carlo sur les données boursières pour prédire les prix futurs et affichez les résultats.
- **Graphiques** : Affichez les résultats de la simulation de Monte Carlo avec des lignes pour le prix actuel et le prix moyen prédit.

### Options
- **Entrée des Données** : Saisissez le symbole boursier et la date d'expiration des options.
- **Affichage des Options** : Affichez les données sur les options, y compris la volatilité implicite en fonction du prix d'exercice et du prix du marché.

### Prévision Économique
- **Choix du Pays et API** : Sélectionnez un pays et saisissez une clé API pour accéder aux données économiques.
- **Analyse du PIB** : Affichez les données du PIB avec des moyennes mobiles et fournissez une analyse historique des statistiques économiques.
- **Analyse du Chômage** : Affichez les données sur le taux de chômage avec des moyennes mobiles et fournissez une analyse historique des statistiques sur le chômage.
- **Analyse de l'Inflation** : Affichez les données sur l'inflation avec des moyennes mobiles et fournissez une analyse historique des statistiques sur l'inflation.

### Obligations
- **Affichage des Obligations** : Consultez les rendements des obligations et affichez les graphiques correspondants.

### Contrats à Terme (Futures)
- **Analyse des Contrats à Terme** : Consultez les données sur les contrats à terme et affichez des graphiques pertinents.

### FOREX
- **Analyse du FOREX** : Analysez les taux de change et affichez des graphiques des paires de devises.

### Carte des Marchés
- **Affichage de la Carte** : Affichez une carte interactive des marchés financiers pour visualiser les performances des différents secteurs.

## Installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/votre-utilisateur/nom-du-repository.git
   pip install -r requirements.txt
   streamlit run app.py

## Contribuer
Les contributions sont les bienvenues ! Veuillez soumettre une pull request ou ouvrir une issue pour discuter des changements que vous souhaitez apporter.

## Licence
Ce projet est sous licence MIT. Veuillez consulter le fichier LICENSE pour plus de détails.