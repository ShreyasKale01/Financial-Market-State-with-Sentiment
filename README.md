# Financial Market State with Sentiment

A quantitative finance research project for identifying and characterizing financial market states by combining market-price relationships with financial-news sentiment.

## Overview

Financial markets do not behave uniformly over time. The objective of this project is to construct market-state representations from asset relationships and investigate whether incorporating news sentiment provides additional information about changing market conditions.

```text
Market Data ───────────────┐
                           ├──> Correlation Structure ──> Market Representation
                           │                                      │
News Data ─> Sentiment ────┘                                      ↓
                                                            Market States
                                                                 ↓
                                                     State Analysis / Prediction
```

## Core Components

### 1. Market Data

Historical price data is transformed into returns and used to study relationships between financial assets.

### 2. Correlation Structure

Correlation matrices are constructed from return series to capture how assets move relative to one another. These matrices provide the basis for studying changes in market structure.

### 3. Market-State Representation

Dimensionality-reduction and clustering techniques can be used to transform high-dimensional correlation information into interpretable market-state representations.

### 4. News Sentiment

Financial news is processed using NLP methods, with FinBERT used to obtain finance-specific sentiment representations from news text.

### 5. Temporal Analysis

Market structure and sentiment are aligned over time to investigate how sentiment and market states evolve together.

## Methods

- Return calculation and financial time-series processing
- Correlation matrices
- Distance transformations
- Multidimensional Scaling (MDS)
- Clustering
- Market-state characterization
- Financial NLP
- FinBERT sentiment analysis
- LSTM-based sequential modeling
- Statistical evaluation and visualization

## Project Structure

```text
Financial-Market-State-with-Sentiment/
├── data/
├── notebooks/
│   ├── 01_market_data.ipynb
│   ├── 02_correlation_analysis.ipynb
│   ├── 03_market_states.ipynb
│   └── 04_sentiment_analysis.ipynb
├── src/
│   ├── data_loader.py
│   ├── returns.py
│   ├── correlation.py
│   ├── market_state.py
│   ├── sentiment.py
│   └── modeling.py
├── results/
│   ├── figures/
│   └── tables/
├── .gitignore
├── requirements.txt
└── README.md
```

## Research Direction

The project studies the relationship between two complementary views of the market:

- **Market structure:** how assets move with one another.
- **Market sentiment:** how financial news reflects positive, negative, or neutral information.

The combination is intended to provide a richer representation of changing market conditions than either source alone.

## Technologies

Python, NumPy, Pandas, SciPy, Scikit-learn, Matplotlib, Seaborn, PyTorch, Hugging Face Transformers, FinBERT, Jupyter.
