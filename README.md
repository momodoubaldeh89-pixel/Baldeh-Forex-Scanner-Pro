# Baldeh_Forex Scanner Pro

A separate Streamlit dashboard combining the original US stock-sector/theme/stock relative-strength scanner with a rule-based forex momentum scanner.

## Files
- `app.py` — Streamlit web app
- `Baldeh_Forex_scanner.py` — stock relative-strength calculations and curated universe
- `requirements.txt` — Python dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy as a separate Streamlit Community Cloud app
1. Create a **new GitHub repository** (do not overwrite your existing scanner repository).
2. Upload all three files from this folder to the repository root.
3. In Streamlit Community Cloud, create a new app and select that repository.
4. Set **Main file path** to `app.py`.
5. Deploy and test the new app on mobile Chrome.

## Notes
- The app retrieves market data from Yahoo Finance via `yfinance`.
- Data may be delayed, incomplete, or unavailable; the app caches data for 15 minutes.
- Forex labels are basic technical conditions and are not guaranteed trade signals.
- The app does not place trades and is not financial advice.
