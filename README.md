# M.A.R.K.T. — Streamlit Trading Analyse


Diese Repo enthält eine einfache Streamlit-Web-App für Trading-Analysen (technische Indikatoren + kurze Fundamental-Analyse via RSS-News).


## Dateien

- `app.py` — die komplette Streamlit-App (einzelne Datei)

- `requirements.txt` — Python-Abhängigkeiten


## Deployment (GitHub → Streamlit Cloud)

1. Erstelle ein neues Repository auf GitHub und pushe `app.py` und `requirements.txt` (und `README.md`).

2. Gehe zu https://streamlit.io/cloud und logge dich ein.

3. Klicke auf "New App" → wähle dein GitHub-Repo und Branch aus → setze `Main file` auf `app.py` → Deploy.


## Hinweise zur Einrichtung auf dem Handy (z.B. Samsung Galaxy S24 Ultra)

1. Öffne den Browser (Chrome oder Samsung Internet) und gehe zur URL deiner Streamlit-App (z.B. `https://<your-app>.streamlit.app`).

2. Um mobile Bedienbarkeit zu verbessern: nutze Hochformat, der Code ist bewusst einfach gehalten, damit Buttons und Eingabefelder groß genug dargestellt werden.

3. Zum schnellen Zugriff: Füge die Seite zum Home-Bildschirm (Browser-Menü → "Zum Startbildschirm hinzufügen").


## Verwendung

- Öffne die App, gib das Symbol (z.B. `AAPL` oder `BTC-USD`) ein, wähle einen Zeitrahmen und drücke "Analyse starten".

- Standard-Passwort zum Login: `LuisAngelitoO` (du kannst das in `app.py` ändern).


## Limitierungen & Tipps

- News-Analyse verwendet öffentliche RSS-Feeds; Qualität und Verfügbarkeit sind nicht garantiert.

- `yfinance` ist praktisch, aber für sehr niedrige Intervalle (1m) oder für bestimmte Assets können Einschränkungen gelten.

- Diese App ist als Lern- und Hilfs-Tool gedacht, keine Anlageberatung.


## Verbesserungen

- API-basierte News (z.B. NewsAPI.org) mit Key für bessere Coverage.

- Ergänzung von Backtesting und Positionsgrößen-Management.

- Integration sicherer Auth (z.B. OAuth / Streamlit-Auth-Providers) statt hartcodiertem Passwort.
