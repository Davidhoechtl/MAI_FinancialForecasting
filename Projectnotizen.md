# Projektbeschreibung

Ich habe einen Prototypen entwickelt, welcher Zeitungsartikel vom Internet scraped und diese mittels **FinBERT** klassifiziert, um das **Market Sentiment** zu analysieren.  

Ich möchte meine Masterarbeit über das Thema **Financial Forecasting in AI** machen. Das Ziel soll sein zu untersuchen, ob ein rein stochastischer Ansatz durch Market Sentiment verbessert werden kann.  

Am Ende soll es möglich sein, unterschiedliche Modelle anhand definierter Metriken zu vergleichen.  

---

## Eingrenzende Punkte für das Thema

1. Bezug auf globale Medien, überwiegend englische Finanzportale (Reuters, Financial Times, Wallstreet Journal, …)
2. Verwendung von tagesaggregierten Sentiments zur Analyse der Markstimmung
3. Erste Evaluierung des Market Sentiment mittels FinBERT zeigt einen Trend, der als valide empfunden wird
4. Prognose großer Aktienindizes: **MSCI World**, **S&P500**
5. Ziel: Kursrichtung vorhersagen (in Prozent) bzw. Klassifizierung von **Up** und **Down**
6. Stochastischer Ansatz: Modelle wie Moving Average oder andere technische Indikatoren (nicht endgültig festgelegt)
7. **ARMA-Modelle** als PoC, später Erweiterung auf:
   - **RNN**, **LSTM**, **Transformer**
   - Standard-ML-Modelle: Multiple Logistic Regression, Linear Regression, XGBoost
8. Performance-Validierung: **Mean Percentage Error**, **Accuracy**, **F1-Score**
9. Kernfrage: Welchen Mehrwert bietet **Sentiment** bei der Vorhersage von Finanzzeitreihen?

---

## Informationen aus Papers

### Investigating the informativeness of technical indicators and news sentiment in financial market price prediction
- Kombination aus technischen Indikatoren und Market Sentiment performt besser als einzeln
- Feature-Extraktion sollte aus einer **Time Series** erfolgen, RNN besser als einfache Regression
- Granger-Causality Test oder Analyse der Modellgewichte zur Feature-Bewertung
- Verkettete LSTMs zur Steigerung der Modelltiefe
- LSTM gut für Feature-Extraktion aus Zeitreihen

### Prediction of stock values changes using sentiment analysis of stock news headlines
- Tools für Sentimentanalyse: TextBlock, VADER, RNN, Transformer Modelle
- Große Unterschiede in der Performance – Vergleich der Methoden möglich

### Sentiment Analysis for Stock Price Prediction - Rubi Gupta, Min Chen
- Yahoo Finance als Datenquelle für Stock Prices
- Analyse muss vorsichtig mit **Lookahead** durchgeführt werden (nur vergangene Nachrichten verwenden)
- Einfluss des Sentiments nimmt mit Zeitfenster zu Tage ab
- Accuracy-Steigerung: ca. 1-3%

### Evaluation of Sentiment Analysis in Finance: From Lexicons to Transformers
- Finanzwelt benötigt spezialisiertes Sentiment-Modell (z.B. Bull, Bear)
- Methoden:
  - Lexikon: Polarity Score
  - Statistische Methoden: TFIDF
  - Word Encoders: Word2Vec, GloVe
  - Sentence / Doc Encoders
  - Transformers: BERT, FinBERT

---

## Datasets
- Pre-labeled Sentiment Datasets: [HuggingFace: financial-news-dataset](https://huggingface.co/datasets/luckycat37/financial-news-dataset)

---

## Informationen aus Experimenten
- S&P500 scheint effizienter Markt zu sein
- ARIMA vs. ARIMA + Sentiment zeigt nur mit **Lookahead (-1)** signifikanten Performance-Boost
- Kurzlebige Signale schnell im Preis enthalten
- Zeitzonen beachten (Headlines UTC, Preise US Eastern UTC-4)

---

## Meeting Notizen

### 29.05.2025
- Weniger Deutschland-spezifisch → globaler Ansatz
- Englische Medien verwenden
- Workflow durchdenken
- Literaturrecherche durchführen

### 19.09.2025
- Bestehende Datasets nutzen, optional Scraper hinzufügen
- Workflow:
  1. Finanzdaten via Yahoo Finance → technische Indikatoren extrahieren
  2. Sentimentdaten via Dataset/Scraper → Sentiment Score berechnen
  3. Daten fusionieren → Feature Matrix
  4. Modelle benchmarken (Regression, XGBoost, RNN, LSTM, LLMs)
- Unterschiedliche Sentiment-Scores je nach Region möglich
- Minimum Fall: ARIMA vs ARIMA + Sentiment

### 03.10.2025
- Grid Search für ARIMA + Sentiment
- Berechnung von unterschiedlichen Sentiment Scores
- Vergleich unterschiedlicher Modelle
- Regionalität als Feature
- Ziel: Proposal, Literaturrecherche zu STOA in Sentimentanalyse, ggf. Vergleich unterschiedlicher Sentiment-Analyse-Methoden, Refactoring

---

## Ziel bis 10.10
- Proposal erstellen
- Literaturrecherche fertigstellen


## 17.10.2025
pipeline refactorn
mehr daten
Sentiment Lag
Sentiment Modelle

Weighted mean für daily sentiment
Sentiment score je quelle
Sentiment LLM -> Gewichtung der Headlines (Relevanz Score)
Regionales Sentiment Feature