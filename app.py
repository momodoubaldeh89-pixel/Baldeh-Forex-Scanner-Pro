from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import Baldeh_Forex_scanner as scanner

st.set_page_config(page_title="Baldeh_Forex Scanner Pro", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1500px}
h1{font-size:clamp(1.6rem,4vw,2.5rem)!important}
h2{font-size:clamp(1.15rem,3vw,1.8rem)!important}
[data-testid="stMetric"]{background:rgba(128,128,128,.08);padding:.65rem;border-radius:.6rem}
.small-note{font-size:.85rem;opacity:.8}
</style>
""", unsafe_allow_html=True)

st.title("📈 Baldeh_Forex Scanner Pro")
st.caption("Combined Forex + US stock-market research dashboard • Mobile-friendly • Educational use")
st.info("Market data is supplied by Yahoo Finance through yfinance and may be delayed, incomplete, or unavailable. Signals are rule-based research indicators, not predictions or financial advice.")

FOREX_PAIRS = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X",
    "USD/CHF": "CHF=X", "AUD/USD": "AUDUSD=X", "USD/CAD": "CAD=X",
    "NZD/USD": "NZDUSD=X", "EUR/GBP": "EURGBP=X", "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X", "AUD/JPY": "AUDJPY=X", "EUR/CHF": "EURCHF=X",
    "USD/CNH": "CNH=X", "USD/SEK": "SEK=X", "USD/NOK": "NOK=X",
}

@st.cache_data(ttl=900, show_spinner=False)
def get_stock_scan():
    all_tickers = [scanner.BENCHMARK] + list(scanner.SECTORS.values()) + [t[2] for t in scanner.THEMES] + [t[0] for t in scanner.STOCKS]
    prices = scanner.download_prices(all_tickers)
    if scanner.BENCHMARK not in prices.columns or prices[scanner.BENCHMARK].dropna().empty:
        raise RuntimeError("Benchmark SPY data is missing.")
    bench = prices[scanner.BENCHMARK]
    bench_returns = {label: scanner.total_return(bench, days) for label, days in scanner.TRADING_WINDOWS.items()}
    sector_map = {ticker: (sector, "Sector ETF") for sector, ticker in scanner.SECTORS.items()}
    sectors = scanner.add_labels(scanner.make_metrics(prices, list(scanner.SECTORS.values()), bench_returns), sector_map)
    if not sectors.empty:
        sectors["Sector"] = sectors["Ticker"].map({ticker: name for name, ticker in scanner.SECTORS.items()})
        sectors = sectors.sort_values("RS Score", ascending=False, na_position="last")
    theme_base = scanner.make_metrics(prices, [t[2] for t in scanner.THEMES], bench_returns)
    theme_lookup = {r["Ticker"]: r.to_dict() for _, r in theme_base.iterrows()} if not theme_base.empty else {}
    theme_rows = []
    for theme, sector, proxy in scanner.THEMES:
        if proxy in theme_lookup:
            row = dict(theme_lookup[proxy]); row.update({"Sector": sector, "Theme": theme, "Proxy ETF": proxy}); theme_rows.append(row)
    themes = pd.DataFrame(theme_rows)
    if not themes.empty: themes = themes.sort_values("RS Score", ascending=False, na_position="last")
    stock_map = {ticker: (sector, theme) for ticker, sector, theme in scanner.STOCKS}
    stocks = scanner.add_labels(scanner.make_metrics(prices, [t[0] for t in scanner.STOCKS], bench_returns), stock_map)
    if not stocks.empty: stocks = stocks.sort_values("RS Score", ascending=False, na_position="last")
    return sectors, themes, stocks, prices.index[-1].date().isoformat(), len(prices.columns)

@st.cache_data(ttl=900, show_spinner=False)
def get_forex_scan(period="1y"):
    tickers = list(FOREX_PAIRS.values())
    raw = yf.download(tickers=tickers, period=period, interval="1d", auto_adjust=True, group_by="column", threads=True, progress=False)
    if raw is None or raw.empty: raise RuntimeError("No forex data returned by Yahoo Finance.")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0): prices = raw["Close"].copy()
        else: prices = raw.xs("Close", axis=1, level=-1)
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    prices.columns = [str(c) for c in prices.columns]
    rows=[]
    for pair, ticker in FOREX_PAIRS.items():
        if ticker not in prices.columns: continue
        s=prices[ticker].dropna()
        if len(s)<55: continue
        last=float(s.iloc[-1]); ma20=float(s.rolling(20).mean().iloc[-1]); ma50=float(s.rolling(50).mean().iloc[-1])
        ret5=(last/float(s.iloc[-6])-1)*100 if len(s)>5 else np.nan
        ret21=(last/float(s.iloc[-22])-1)*100 if len(s)>21 else np.nan
        ret63=(last/float(s.iloc[-64])-1)*100 if len(s)>63 else np.nan
        delta=s.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean()
        rs=gain.iloc[-1]/loss.iloc[-1] if pd.notna(loss.iloc[-1]) and loss.iloc[-1] != 0 else np.nan
        rsi=float(100-(100/(1+rs))) if pd.notna(rs) else (100.0 if pd.notna(gain.iloc[-1]) and gain.iloc[-1]>0 else np.nan)
        up=last>ma20>ma50; down=last<ma20<ma50
        if up and rsi < 70: signal="BULLISH MOMENTUM"
        elif down and rsi > 30: signal="BEARISH MOMENTUM"
        elif up: signal="UPTREND — EXTENDED RSI"
        elif down: signal="DOWNTREND — LOW RSI"
        else: signal="MIXED / WAIT"
        rows.append({"Pair":pair,"Yahoo Symbol":ticker,"Last Close":last,"1W Change %":ret5,"1M Change %":ret21,"3M Change %":ret63,"20D MA":ma20,"50D MA":ma50,"RSI (14)":rsi,"Signal":signal,"As Of":s.index[-1].date().isoformat()})
    df=pd.DataFrame(rows)
    if not df.empty: df=df.sort_values(["1M Change %","3M Change %"],ascending=False,na_position="last")
    return df, prices.index[-1].date().isoformat()

def show_table(df, cols=None):
    if df is None or df.empty:
        st.warning("No usable data returned for this section."); return
    out=df.copy()
    if cols: out=out[[c for c in cols if c in out.columns]]
    st.dataframe(out, use_container_width=True, hide_index=True)

with st.sidebar:
    st.header("Scan controls")
    if st.button("🔄 Refresh all market data", type="primary", use_container_width=True):
        get_stock_scan.clear(); get_forex_scan.clear(); st.session_state["force_refresh_notice"] = True
    st.caption("Cached data refreshes every 15 minutes. Use the refresh button to request new data.")

try:
    with st.spinner("Loading stock, sector, theme and forex data…"):
        sectors, themes, stocks, stock_date, stock_symbols = get_stock_scan()
        forex, forex_date = get_forex_scan()
except Exception as exc:
    st.error(f"The scan could not be completed: {exc}")
    st.info("Check the app logs, internet access, package installation and Yahoo Finance availability. Try refreshing later.")
    st.stop()

st.caption(f"Stock data through: **{stock_date}** · Forex data through: **{forex_date}** · Stock symbols returned: **{stock_symbols}** · App updated: {datetime.now().astimezone():%Y-%m-%d %H:%M %Z}")

# Summary cards
m1,m2,m3,m4=st.columns(4)
m1.metric("Top sector", str(sectors.iloc[0]["Sector"]) if not sectors.empty else "N/A")
m2.metric("Top theme", str(themes.iloc[0]["Theme"]) if not themes.empty else "N/A")
m3.metric("Top stock", str(stocks.iloc[0]["Ticker"]) if not stocks.empty else "N/A")
bullish=int(forex["Signal"].astype(str).str.contains("BULLISH").sum()) if not forex.empty else 0
m4.metric("Bullish forex setups", bullish)

tab_fx,tab_sector,tab_theme,tab_stock,tab_method=st.tabs(["💱 Forex Scanner","🏆 Sectors","🔥 Themes","📈 Stocks","ℹ️ Method & Downloads"])
with tab_fx:
    st.subheader("Forex pair momentum scanner")
    st.caption("Returns are price changes for the quoted pair, not a guarantee of a trade. For USD/JPY and similar inversely quoted pairs, interpret the pair direction carefully.")
    if not forex.empty:
        choices=["All signals"]+sorted(forex["Signal"].dropna().unique().tolist())
        chosen=st.selectbox("Filter by signal",choices,key="fx_signal_filter")
        fview=forex if chosen=="All signals" else forex[forex["Signal"]==chosen]
        show_table(fview,["Pair","Last Close","1W Change %","1M Change %","3M Change %","RSI (14)","Signal","As Of"])
        st.download_button("Download forex CSV",forex.to_csv(index=False).encode("utf-8"),"forex_scan.csv","text/csv",use_container_width=True)
    else: st.warning("No forex symbols returned enough usable history.")
with tab_sector:
    st.subheader("All 11 US market sectors")
    show_table(sectors,["RS Rank","Sector","Ticker","RS Score","1M SPY Excess %","3M SPY Excess %","6M SPY Excess %","12M SPY Excess %","Above 50D","Above 200D","50D Above 200D","Signal"])
    st.download_button("Download sector CSV",sectors.to_csv(index=False).encode("utf-8"),"sector_rankings.csv","text/csv",use_container_width=True)
with tab_theme:
    st.subheader("Investment theme relative strength")
    if not themes.empty:
        sectors_list=["All sectors"]+sorted(themes["Sector"].dropna().unique().tolist())
        chosen_sector=st.selectbox("Filter themes by sector",sectors_list,key="theme_sector")
        tview=themes if chosen_sector=="All sectors" else themes[themes["Sector"]==chosen_sector]
        show_table(tview,["RS Rank","Theme","Sector","Proxy ETF","RS Score","1M SPY Excess %","3M SPY Excess %","6M SPY Excess %","12M SPY Excess %","Signal"])
    else: st.warning("No theme data returned.")
    st.download_button("Download themes CSV",themes.to_csv(index=False).encode("utf-8"),"theme_rankings.csv","text/csv",use_container_width=True)
with tab_stock:
    st.subheader("Individual stock relative strength")
    if not stocks.empty:
        c1,c2=st.columns(2)
        with c1: chosen_sector=st.selectbox("Filter stocks by sector",["All sectors"]+sorted(stocks["Sector"].dropna().unique().tolist()),key="stock_sector")
        with c2: search=st.text_input("Search ticker or theme",placeholder="e.g. NVDA, cybersecurity")
        view=stocks if chosen_sector=="All sectors" else stocks[stocks["Sector"]==chosen_sector]
        if search:
            q=search.lower(); view=view[view["Ticker"].astype(str).str.lower().str.contains(q,na=False)|view["Theme"].astype(str).str.lower().str.contains(q,na=False)]
        if st.checkbox("Show only positive RS score",value=False): view=view[view["RS Score"]>0]
        show_table(view,["RS Rank","Ticker","Sector","Theme","RS Score","1M SPY Excess %","3M SPY Excess %","6M SPY Excess %","12M SPY Excess %","Above 50D","Above 200D","50D Above 200D","Signal"])
    else: st.warning("No stock data returned.")
    st.download_button("Download stocks CSV",stocks.to_csv(index=False).encode("utf-8"),"stock_rankings.csv","text/csv",use_container_width=True)
with tab_method:
    st.subheader("How the scanners work")
    st.markdown("""
    **US stocks, sectors and themes**
    - Benchmark: SPY, used as an S&P 500 ETF proxy.
    - Relative-strength score: weighted excess returns versus SPY — 1 month 20%, 3 months 30%, 6 months 30%, 12 months 20%.
    - Trend checks: price above 50-day average, price above 200-day average, and 50-day average above 200-day average.
    - The stock universe is a curated watchlist, not every listed company. Theme ETFs can overlap.

    **Forex**
    - Scans a curated set of major and selected cross-currency pairs using daily price data.
    - Displays 1-week, 1-month and 3-month price changes, 20-day/50-day moving averages, and RSI(14).
    - Bullish/bearish labels are simple rule-based momentum conditions, not validated predictive signals. They do not include spread, slippage, fees, macro events, or your account-specific risk.

    **Data and risk**
    - Data: Yahoo Finance through yfinance; it may be delayed, incomplete, corrected, or temporarily unavailable.
    - This app does not place trades. Verify all information independently and test any strategy before risking capital.
    """)

st.caption("Baldeh_Forex Scanner Pro • Research and education only • No trades are placed.")
