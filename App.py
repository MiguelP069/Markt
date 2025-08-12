# app.py

# Streamlit Trading-Analyse App (einzelne Datei)

# Passwortschutz: 'LuisAngelitoO' (wie vom Nutzer gewünscht)


import streamlit as st

import yfinance as yf

import pandas as pd

import numpy as np

import feedparser

import datetime

import plotly.graph_objects as go

from plotly.subplots import make_subplots


# technische Indikatoren: versuche 'ta' (pip install ta). Falls nicht vorhanden, berechnen wir einfache Varianten.

try:

  import ta

    TA_AVAILABLE = True

except Exception:

    TA_AVAILABLE = False


st.set_page_config(page_title='M.A.R.K.T. — Trading Analyse', layout='centered')


# ---------- Helper functions ----------


def check_password():

    """Einfacher Passwort-Check, speichert Login in session_state."""

    if 'authenticated' not in st.session_state:

        st.session_state.authenticated = False


    if st.session_state.authenticated:

        return True


    with st.form('login_form'):

        pwd = st.text_input('Passwort', type='password')

        submitted = st.form_submit_button('Einloggen')

        if submitted:

            if pwd == 'LuisAngelitoO':

                st.session_state.authenticated = True

                st.experimental_rerun()

            else:

                st.error('Falsches Passwort.')

    return False



def map_timeframe_to_yfinance(timeframe: str):

    """Mappt eine vom Benutzer gewählte Periode auf yfinance period+interval.

    Rückgabe: (period, interval)

    """

    mapping = {

        '1h': ('7d', '1h'),

        '4h': ('30d', '4h'),

        '1d': ('1y', '1d'),

        '1W': ('5y', '1wk'),

        '1M': ('10y', '1mo')

    }

    return mapping.get(timeframe, ('1y', '1d'))



def fetch_data(symbol: str, period: str, interval: str):

    """Lädt historische Daten via yfinance.

    Gibt DataFrame mit DatetimeIndex und OHLCV zurück.

    """

    try:

        ticker = yf.Ticker(symbol)

        df = ticker.history(period=period, interval=interval, actions=False)

        if df.empty:

            raise ValueError('Keine Daten gefunden. Überprüfe das Symbol oder den Zeitrahmen.')

        df = df.dropna()

        return df

    except Exception as e:

        raise



# Indikatoren (Fallback-Implementationen) ------------------------------------------------


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:

    d = df.copy()

    # Gleitende Durchschnitte

    d['SMA_20'] = d['Close'].rolling(window=20, min_periods=1).mean()

    d['SMA_50'] = d['Close'].rolling(window=50, min_periods=1).mean()

    d['SMA_200'] = d['Close'].rolling(window=200, min_periods=1).mean()

    d['EMA_20'] = d['Close'].ewm(span=20, adjust=False).mean()


    # RSI

    try:

        if TA_AVAILABLE:

            d['rsi'] = ta.momentum.RSIIndicator(d['Close'], window=14).rsi()

        else:

            delta = d['Close'].diff()

            up = delta.clip(lower=0)

            down = -1 * delta.clip(upper=0)

            ma_up = up.rolling(window=14, min_periods=1).mean()

            ma_down = down.rolling(window=14, min_periods=1).mean()

            rs = ma_up / (ma_down + 1e-9)

            d['rsi'] = 100 - (100 / (1 + rs))

    except Exception:

        d['rsi'] = np.nan


    # MACD

    try:

        if TA_AVAILABLE:

            macd = ta.trend.MACD(d['Close'])

            d['macd'] = macd.macd()

            d['macd_signal'] = macd.macd_signal()

        else:

            ema12 = d['Close'].ewm(span=12, adjust=False).mean()

            ema26 = d['Close'].ewm(span=26, adjust=False).mean()

            d['macd'] = ema12 - ema26

            d['macd_signal'] = d['macd'].ewm(span=9, adjust=False).mean()

    except Exception:

        d['macd'] = np.nan

        d['macd_signal'] = np.nan


    # Stochastic Oscillator

    try:

        low14 = d['Low'].rolling(14, min_periods=1).min()

        high14 = d['High'].rolling(14, min_periods=1).max()

        d['stoch_%K'] = 100 * (d['Close'] - low14) / (high14 - low14 + 1e-9)

        d['stoch_%D'] = d['stoch_%K'].rolling(3, min_periods=1).mean()

    except Exception:

        d['stoch_%K'] = np.nan

        d['stoch_%D'] = np.nan


    # ATR (für Stop-Loss Vorschlag)

    try:

        high_low = d['High'] - d['Low']

        high_close = np.abs(d['High'] - d['Close'].shift())

        low_close = np.abs(d['Low'] - d['Close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        d['ATR_14'] = tr.rolling(14, min_periods=1).mean()

    except Exception:

        d['ATR_14'] = np.nan


    return d



def find_support_resistance(df: pd.DataFrame, n=5):

    """Einfache Methode: finde lokale Maxima/Minima in einem Rolling-Fenster als potenzielle S/R.

    Gibt zwei Listen: supports, resistances (jeweils Werte)

    """

    prices = df['Close']

    supports = []

    resistances = []

    for i in range(n, len(prices) - n):

        window = prices.iloc[i - n:i + n + 1]

        center = prices.iloc[i]

        if center == window.min():

            supports.append((prices.index[i], center))

        if center == window.max():

            resistances.append((prices.index[i], center))

    # Nimm die stärksten (höchsten / niedrigsten) 3 Levels

    supports_sorted = sorted(supports, key=lambda x: x[1])[:3]

    resistances_sorted = sorted(resistances, key=lambda x: x[1], reverse=True)[:3]

    return supports_sorted, resistances_sorted



# Fundamental: News via RSS (feedparser)


def fetch_news_rss(symbol: str, max_items=10):

    """Versucht, ticker-spezifische RSS-Feeds zu lesen (z.B. Yahoo Finance). Falls leer, fallback auf Top-Finance-Feed."""

    feeds_to_try = [

        f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US',

        'https://finance.yahoo.com/rss/topstories',

        'https://www.reuters.com/finance',

        'https://www.reuters.com/markets/wealth/rss.xml'

    ]

    entries = []

    for f in feeds_to_try:

        try:

            parsed = feedparser.parse(f)

            if parsed.bozo:

                # bozo signals parse issue, but still may have entries

                pass

            for e in parsed.entries[:max_items]:

                title = e.get('title', '')

                summary = e.get('summary', '')

                link = e.get('link', '')

                published = e.get('published', '')

                entries.append({'title': title, 'summary': summary, 'link': link, 'published': published, 'source': f})

            if entries:

                break

        except Exception:

            continue

    return entries



def simple_sentiment_from_news(entries):

    """Sehr einfache Heuristik: positive/negative Wortlisten.

    Liefert score in [-1,1] und kurze Zusammenfassung.

    """

    if not entries:

        return 0.0, 'Keine News gefunden.'

    pos_words = ['beat', 'beats', 'surge', 'gain', 'record', 'raise', 'upgrade', 'buy', 'positive', 'profit', 'growth']

    neg_words = ['fall', 'falls', 'drop', 'loss', 'miss', 'downgrade', 'sell', 'negative', 'decline', 'warn']

    score = 0

    snippets = []

    for e in entries:

        txt = (e['title'] + ' ' + e['summary']).lower()

        p = sum(txt.count(w) for w in pos_words)

        n = sum(txt.count(w) for w in neg_words)

        score += (p - n)

        snippets.append((e['title'], e.get('link', '')))

    # normalize

    if score == 0:

        return 0.0, 'Neutral oder keine starke Sentiment-Signale.'

    norm = np.tanh(score / max(1, len(entries)))

    return float(norm), snippets[:5]



def make_recommendation(df: pd.DataFrame, news_score: float):

    """Einfache Regel-basierte Empfehlung: BUY / SELL / HOLD.

    Kombiniert technische Indikatoren und Nachrichten-Sentiment.

    """

    latest = df.iloc[-1]

    score = 0


    # Preis vs MAs

    if latest['Close'] > latest['SMA_50']:

        score += 1

    else:

        score -= 1

    if latest['Close'] > latest['SMA_200']:

        score += 2

    else:

        score -= 2


    # MACD momentum

    if latest['macd'] > latest['macd_signal']:

        score += 1

    else:

        score -= 1


    # RSI

    if latest['rsi'] < 30:

        score += 1  # oversold

    elif latest['rsi'] > 70:

        score -= 1  # overbought


    # Stochastic

    if latest['stoch_%K'] < 20:

        score += 0.5

    elif latest['stoch_%K'] > 80:

        score -= 0.5


    # News influence

    score += np.sign(news_score) * min(2, abs(news_score) * 2)


    # Entscheidung

    if score >= 3:

        rec = 'BUY'

    elif score <= -3:

        rec = 'SELL'

    else:

        rec = 'HOLD'


    return rec, score



def suggest_stop_take(latest_price: float, atr: float, supports, resistances):

    """Basierend auf ATR und nächsten S/R, Vorschlag für Stop-Loss und Take-Profit.

    """

    if np.isnan(atr) or atr == 0:

        atr = latest_price * 0.01  # 1% fallback


    # Stop: etwas unter letztem Support oder price - 1.5*ATR

    if supports:

        nearest_support = supports[-1][1]  # take the highest support from list

        stop = min(latest_price - 1.5 * atr, nearest_support - 0.5 * atr)

    else:

        stop = latest_price - 1.5 * atr


    # Take: nächster Widerstand oder price + 3*ATR

    if resistances:

        nearest_res = resistances[0][1]

        take = max(latest_price + 3 * atr, nearest_res + 0.5 * atr)

    else:

        take = latest_price + 3 * atr


    # Safety: stop nicht über dem aktuellen Preis

    stop = min(stop, latest_price * 0.995)


    return float(max(0, stop)), float(take)



# ---------- UI ----------


if not check_password():

    st.stop()


st.title('M.A.R.K.T. — Trading Analyse')

st.markdown('Führe eine schnelle Trading-Analyse durch (Technisch + Kurz-Fundamental via RSS-News).')


with st.form('inputs'):

    col1, col2 = st.columns([2,1])

    with col1:

        symbol = st.text_input('Asset-Symbol (z.B. AAPL oder BTC-USD)', value='AAPL')

    with col2:

        timeframe = st.selectbox('Zeitrahmen', options=['1h','4h','1d','1W','1M'], index=2)

    submitted = st.form_submit_button('Analyse starten')


if not submitted:

    st.info('Gib ein Symbol ein und drücke "Analyse starten".')

    st.stop()


with st.spinner('Daten werden geladen...'):

    period, interval = map_timeframe_to_yfinance(timeframe)

    try:

        df_raw = fetch_data(symbol, period=period, interval=interval)

    except Exception as e:

        st.error(f'Fehler beim Laden der Daten: {e}')

        st.stop()


# compute indicators

with st.spinner('Indikatoren berechnen...'):

    df = compute_indicators(df_raw)


# find S/R

supports, resistances = find_support_resistance(df, n=5)


# fetch news

with st.spinner('News abrufen (RSS)...'):

    news_entries = fetch_news_rss(symbol)

    news_score, news_snips = simple_sentiment_from_news(news_entries)


# recommendation

rec, rec_score = make_recommendation(df, news_score)

latest = df.iloc[-1]

stop, take = suggest_stop_take(latest['Close'], latest.get('ATR_14', np.nan), supports, resistances)


# ---------- Darstellung ----------


st.subheader(f'Ergebnis: {rec}')

st.metric('Letzter Preis', f"{latest['Close']:.2f}", delta=None)

st.write(f'Empfehlungs-Score: {rec_score:.2f}')

st.write('Vorgeschlagener Stop-Loss und Take-Profit:')

st.write({'Stop-Loss': round(stop, 4), 'Take-Profit': round(take, 4)})


# Charts

with st.expander('Preis & Indikatoren (Chart)'):

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,

                        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.02)


    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA20', line=dict(width=1)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA50', line=dict(width=1)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA200', line=dict(width=1)), row=1, col=1)


    # MACD

    fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD'), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name='MACD Signal'), row=2, col=1)


    # RSI

    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI'), row=3, col=1)

    fig.update_yaxes(title_text='Price', row=1, col=1)

    fig.update_yaxes(title_text='MACD', row=2, col=1)

    fig.update_yaxes(title_text='RSI', row=3, col=1)

    fig.update_layout(height=700, showlegend=True)

    st.plotly_chart(fig, use_container_width=True)


with st.expander('Support & Resistance (gefunden)'):

    st.write('Supports (Datum, Preis):')

    st.write(supports)

    st.write('Resistances (Datum, Preis):')

    st.write(resistances)


with st.expander('Kurz-Analyse (technisch)'):

    st.markdown('**Technische Kernaussagen:**')

    lines = []

    if latest['Close'] > latest['SMA_50']:

        lines.append('- Preis liegt über SMA50 → kurzfristig bullisches Zeichen')

    else:

        lines.append('- Preis liegt unter SMA50 → kurzfristig bärisch')

    if latest['Close'] > latest['SMA_200']:

        lines.append('- Preis liegt über SMA200 → langfristig bullisch')

    else:

        lines.append('- Preis liegt unter SMA200 → langfristig bärisch')

    if latest['macd'] > latest['macd_signal']:

        lines.append('- MACD zeigt bullische Dynamik')

    else:

        lines.append('- MACD zeigt bärische Dynamik')

    lines.append(f"- Aktueller RSI: {latest['rsi']:.1f}")

    st.write('\n'.join(lines))


with st.expander('Kurz-Fundamental (News)'):

    st.markdown('**News-Summary (via RSS)**')

    st.write(f'News-Sentiment-Score (heuristisch): {news_score:.3f}')

    if isinstance(news_snips, str):

        st.write(news_snips)

    else:

        for t, l in news_snips:

            st.write(f'- [{t}]({l})')


with st.expander('Detaillierte Rohdaten'):

    st.dataframe(df.tail(200))


st.markdown('---')

st.caption('Hinweis: Diese App liefert keine Finanzberatung. Prüfe Ergebnisse eigenständig.')
