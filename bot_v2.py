# ======================================================
# SELF-ASSESSMENT S&D SCALPING BOT V2
# ======================================================

import os
import time
import ccxt
import pandas as pd
import threading
from flask import Flask, jsonify
import requests
import logging
from datetime import datetime, timezone
import json
from collections import defaultdict

# ======================================================
# LOGGING SETUP
# ======================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("SDBOT_V2")

# ======================================================
# CONFIG
# ======================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "8465314518:AAELNPpPQPwl9nQfS6KAUqKgVHx-wD7IDlA").strip()

CHAT_ID1 = os.getenv("CHAT_ID", "").strip()
CHAT_ID2 = os.getenv("CHAT_ID2", "").strip()
RAW_CHAT_IDS = os.getenv("CHAT_IDS", "")

CHAT_IDS = set()
if CHAT_ID1:
    CHAT_IDS.add(CHAT_ID1)
if CHAT_ID2:
    CHAT_IDS.add(CHAT_ID2)
if RAW_CHAT_IDS:
    for cid in RAW_CHAT_IDS.split(","):
        cid = cid.strip()
        if cid:
            CHAT_IDS.add(cid)
CHAT_IDS = list(CHAT_IDS)

PORT = int(os.getenv("PORT", 10000))
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 20))
PAIR_LIMIT = int(os.getenv("PAIR_LIMIT", 80))
TOP_MOVER_COUNT = int(os.getenv("TOP_MOVER_COUNT", 12))
WINDOW = int(os.getenv("WINDOW", 1800))

EXCHANGES = ["binance", "binance_futures", "kucoin", "bybit", "okx"]

# Trade tracking settings
TRADE_RESOLUTION_TIMEOUT = 3600  # 60 minutes to resolve trade
ANALYSIS_INTERVAL = 3600  # Analyze every hour

# ======================================================
# DATA STRUCTURES
# ======================================================

recent_signals = {}
active_trades = {}  # tracking live trades
completed_trades = []  # historical trades with outcomes
trade_counter = 0

# ======================================================
# TELEGRAM UTILITIES
# ======================================================

def send_telegram(text: str):
    """Send Telegram messages to ALL configured chat IDs."""
    if not BOT_TOKEN:
        log.error("BOT_TOKEN missing")
        return
    
    if not CHAT_IDS:
        log.warning("No chat IDs configured")
        return
    
    encoded = requests.utils.quote(text)
    
    for cid in CHAT_IDS:
        try:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?chat_id={cid}&text={encoded}&parse_mode=Markdown"
            requests.get(url, timeout=5)
        except Exception as e:
            log.error(f"Telegram error for {cid}: {e}")

def send_startup():
    """Notify chats when bot starts."""
    msg = (
        "🚀 *SELF-ASSESSMENT S&D BOT V2 ACTIVE*\n\n"
        f"Exchanges: {', '.join(EXCHANGES)}\n"
        f"Scan Interval: {SCAN_INTERVAL}s\n"
        f"Pairs per Exchange: {PAIR_LIMIT}\n"
        f"Top Movers: {TOP_MOVER_COUNT}\n\n"
        "🧠 *NEW FEATURES:*\n"
        "• Live trade tracking\n"
        "• Win/Loss analysis\n"
        "• Context-based learning\n"
        "• Performance insights\n\n"
        "Real-time breakout scanner with self-assessment is now running ⚡"
    )
    send_telegram(msg)
    log.info(f"Startup message sent → chats: {CHAT_IDS}")

# ======================================================
# DUPLICATE PROTECTION
# ======================================================

def allow(symbol, direction):
    now = time.time()
    key = f"{symbol}_{direction}"
    
    if symbol not in recent_signals:
        recent_signals[symbol] = {}
    
    if key not in recent_signals[symbol]:
        recent_signals[symbol][key] = now
        return True
    
    if now - recent_signals[symbol][key] > WINDOW:
        recent_signals[symbol][key] = now
        return True
    
    return False

# ======================================================
# INDICATORS (UNCHANGED)
# ======================================================

def add_indicators(df):
    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    
    df["vol_sma"] = df["volume"].rolling(20).mean()
    
    df["atr_raw"] = df["high"] - df["low"]
    df["atr"] = df["atr_raw"].rolling(14).mean()
    df["atr_sma"] = df["atr"].rolling(14).mean()
    
    df["range"] = df["high"] - df["low"]
    return df

def get_df(ex, symbol, tf):
    try:
        data = ex.fetch_ohlcv(symbol, tf, limit=120)
        df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
        return add_indicators(df)
    except Exception as e:
        log.error(f"Fetch error {symbol} {tf}: {e}")
        return None

# ======================================================
# EXCHANGES
# ======================================================

def get_ex(name):
    try:
        if name == "binance_futures":
            return ccxt.binance({"options": {"defaultType": "future"}})
        if name == "bybit":
            return ccxt.bybit({"options": {"defaultType": "linear"}})
        return getattr(ccxt, name)()
    except Exception as e:
        log.error(f"Exchange load error ({name}): {e}")
        return None

def get_pairs(ex):
    try:
        mk = ex.load_markets()
        return [s for s in mk if s.endswith("USDT")][:PAIR_LIMIT]
    except:
        return []

# ======================================================
# TOP MOVERS
# ======================================================

def detect_top_movers(ex):
    movers = []
    pairs = get_pairs(ex)
    
    for s in pairs:
        df = get_df(ex, s, "15m")
        if df is None or len(df) < 20:
            continue
        
        pct_change = (df["close"].iloc[-1] - df["close"].iloc[-4]) / df["close"].iloc[-4] * 100
        vol_ratio = df["volume"].iloc[-1] / (df["vol_sma"].iloc[-1] + 1e-10)
        
        score = pct_change * 0.55 + vol_ratio * 0.45
        movers.append((s, score))
    
    movers_sorted = sorted(movers, key=lambda x: x[1], reverse=True)
    return [m[0] for m in movers_sorted[:TOP_MOVER_COUNT]]

# ======================================================
# TRADING LOGIC (UNCHANGED)
# ======================================================

def trend_long(df5, df15):
    return (
        df5["ema9"].iloc[-1] > df5["ema20"].iloc[-1] > df5["ema50"].iloc[-1] and
        df15["ema9"].iloc[-1] > df15["ema20"].iloc[-1] > df15["ema50"].iloc[-1]
    )

def trend_short(df5, df15):
    return (
        df5["ema9"].iloc[-1] < df5["ema20"].iloc[-1] < df5["ema50"].iloc[-1] and
        df15["ema9"].iloc[-1] < df15["ema20"].iloc[-1] < df15["ema50"].iloc[-1]
    )

def volatility_ok(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    return last["atr"] > last["atr_sma"] and last["atr"] > prev["atr"] * 1.02

def volume_ok(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    return last["volume"] > last["vol_sma"] * 1.7 and last["volume"] > prev["volume"]

def find_recent_swing_high(df):
    for i in range(len(df)-3, 2, -1):
        if df["high"].iloc[i] > df["high"].iloc[i-1] and df["high"].iloc[i] > df["high"].iloc[i+1]:
            return df["high"].iloc[i]
    return None

def find_recent_swing_low(df):
    for i in range(len(df)-3, 2, -1):
        if df["low"].iloc[i] < df["low"].iloc[i-1] and df["low"].iloc[i] < df["low"].iloc[i+1]:
            return df["low"].iloc[i]
    return None

def find_sd_zones(df):
    zones = []
    for i in range(3, len(df)-3):
        base = df.iloc[i]
        prev = df.iloc[i-1]
        nxt = df.iloc[i+1]
        
        if base["close"] > base["open"] and nxt["close"] > nxt["open"] and (nxt["close"] - nxt["open"]) > base["range"] * 1.2:
            zones.append(("demand", base["low"], prev["high"]))
        
        if base["close"] < base["open"] and nxt["close"] < nxt["open"] and (base["open"] - base["close"]) > prev["range"] * 1.2:
            zones.append(("supply", base["high"], prev["low"]))
    
    return zones[-2:]

def in_supply(price, zones):
    for z in zones:
        if z[0] == "supply" and z[2] <= price <= z[1]:
            return True
    return False

def in_demand(price, zones):
    for z in zones:
        if z[0] == "demand" and z[1] <= price <= z[2]:
            return True
    return False

def near_supply(price, zones):
    for z in zones:
        if z[0] == "supply" and abs(price - z[2]) / price < 0.0005:
            return True
    return False

def near_demand(price, zones):
    for z in zones:
        if z[0] == "demand" and abs(price - z[2]) / price < 0.0005:
            return True
    return False

def breakout_long(df5, df15):
    last = df5.iloc[-1]
    price = last["close"]
    p1 = df5.iloc[-2]
    p2 = df5.iloc[-3]
    
    if not trend_long(df5, df15):
        return False
    if not volatility_ok(df5) or not volume_ok(df5):
        return False
    
    swing_high = find_recent_swing_high(df5)
    if swing_high is None or price <= swing_high * 1.0004:
        return False
    
    sd5 = find_sd_zones(df5)
    sd15 = find_sd_zones(df15)
    
    if in_supply(price, sd5) or in_supply(price, sd15):
        return False
    if near_supply(price, sd5) or near_supply(price, sd15):
        return False
    
    breakout = max(p1["high"], p2["high"])
    if not (price > breakout * 1.0004):
        return False
    
    body = last["close"] - last["open"]
    return body > 0 and body >= 0.50 * last["range"]

def breakout_short(df5, df15):
    last = df5.iloc[-1]
    price = last["close"]
    p1 = df5.iloc[-2]
    p2 = df5.iloc[-3]
    
    if not trend_short(df5, df15):
        return False
    if not volatility_ok(df5) or not volume_ok(df5):
        return False
    
    swing_low = find_recent_swing_low(df5)
    if swing_low is None or price >= swing_low * 0.9996:
        return False
    
    sd5 = find_sd_zones(df5)
    sd15 = find_sd_zones(df15)
    
    if in_demand(price, sd5) or in_demand(price, sd15):
        return False
    if near_demand(price, sd5) or near_demand(price, sd15):
        return False
    
    breakdown = min(p1["low"], p2["low"])
    if not (price < breakdown * 0.9996):
        return False
    
    body = last["open"] - last["close"]
    return body > 0 and body >= 0.50 * last["range"]

# ======================================================
# CONTEXT SNAPSHOT (NEW - PHASE 1)
# ======================================================

def get_trading_session():
    """Determine current trading session based on UTC time."""
    hour = datetime.now(timezone.utc).hour
    if 0 <= hour < 8:
        return "Asia"
    elif 8 <= hour < 16:
        return "London"
    else:
        return "NY"

def calculate_trend_strength(df5, df15):
    """Categorize trend strength: weak/medium/strong."""
    ema_sep_5 = abs(df5["ema9"].iloc[-1] - df5["ema50"].iloc[-1]) / df5["close"].iloc[-1]
    ema_sep_15 = abs(df15["ema9"].iloc[-1] - df15["ema50"].iloc[-1]) / df15["close"].iloc[-1]
    
    avg_sep = (ema_sep_5 + ema_sep_15) / 2
    
    if avg_sep > 0.015:
        return "strong"
    elif avg_sep > 0.008:
        return "medium"
    else:
        return "weak"

def calculate_volume_category(df):
    """Categorize volume expansion: normal/high/extreme."""
    last_vol = df["volume"].iloc[-1]
    vol_sma = df["vol_sma"].iloc[-1]
    ratio = last_vol / (vol_sma + 1e-10)
    
    if ratio > 2.5:
        return "extreme"
    elif ratio > 1.7:
        return "high"
    else:
        return "normal"

def calculate_atr_state(df):
    """Categorize ATR state: rising/flat/spiking."""
    atr_curr = df["atr"].iloc[-1]
    atr_prev = df["atr"].iloc[-2]
    atr_sma = df["atr_sma"].iloc[-1]
    
    change = (atr_curr - atr_prev) / (atr_prev + 1e-10)
    
    if atr_curr > atr_sma * 1.15 and change > 0.05:
        return "spiking"
    elif change > 0.02:
        return "rising"
    else:
        return "flat"

def get_pair_category(symbol):
    """Categorize pair type."""
    if "BTC" in symbol or "ETH" in symbol:
        return "major"
    elif any(x in symbol for x in ["SOL", "AVAX", "LINK", "BNB", "ADA", "DOT", "MATIC"]):
        return "large_cap"
    else:
        return "micro_cap"

def calculate_sl_atr_ratio(sl_distance, atr):
    """Calculate SL distance in ATR multiples."""
    ratio = sl_distance / (atr + 1e-10)
    
    if ratio < 0.5:
        return "very_tight"
    elif ratio < 1.0:
        return "tight"
    elif ratio < 1.5:
        return "medium"
    else:
        return "wide"

def capture_context(symbol, direction, entry, sl, atr, df5, df15):
    """Capture comprehensive trading context for learning."""
    sl_distance = abs(entry - sl)
    
    context = {
        "trend_strength": calculate_trend_strength(df5, df15),
        "volume_category": calculate_volume_category(df5),
        "atr_state": calculate_atr_state(df5),
        "session": get_trading_session(),
        "pair_category": get_pair_category(symbol),
        "sl_atr_ratio": calculate_sl_atr_ratio(sl_distance, atr),
        "atr_value": float(atr),
        "atr_zero": atr < 1e-8
    }
    
    return context

# ======================================================
# TRADE TRACKING (NEW - PHASE 1)
# ======================================================

def create_trade(symbol, direction, entry, sl, tp1, tp2, tp3, tp4, atr, context, exchange):
    """Create a new tracked trade."""
    global trade_counter
    trade_counter += 1
    
    trade = {
        "id": trade_counter,
        "symbol": symbol,
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "tp4": tp4,
        "atr": atr,
        "exchange": exchange,
        "context": context,
        "timestamp": time.time(),
        "outcome": "PENDING",
        "resolution_time": None,
        "hit_level": None
    }
    
    active_trades[trade_counter] = trade
    log.info(f"Trade #{trade_counter} created: {symbol} {direction}")
    
    return trade

def check_trade_outcome(trade, ex):
    """Check if trade has hit TP1 or SL."""
    try:
        symbol = trade["symbol"]
        df = get_df(ex, symbol, "5m")
        
        if df is None or len(df) == 0:
            return False
        
        current_price = df["close"].iloc[-1]
        high = df["high"].iloc[-1]
        low = df["low"].iloc[-1]
        
        if trade["direction"] == "LONG":
            # Check TP1 hit
            if high >= trade["tp1"]:
                trade["outcome"] = "WIN"
                trade["hit_level"] = "TP1"
                trade["resolution_time"] = time.time()
                return True
            # Check SL hit
            elif low <= trade["sl"]:
                trade["outcome"] = "LOSS"
                trade["hit_level"] = "SL"
                trade["resolution_time"] = time.time()
                return True
        
        else:  # SHORT
            # Check TP1 hit
            if low <= trade["tp1"]:
                trade["outcome"] = "WIN"
                trade["hit_level"] = "TP1"
                trade["resolution_time"] = time.time()
                return True
            # Check SL hit
            elif high >= trade["sl"]:
                trade["outcome"] = "LOSS"
                trade["hit_level"] = "SL"
                trade["resolution_time"] = time.time()
                return True
        
        return False
        
    except Exception as e:
        log.error(f"Error checking trade outcome: {e}")
        return False

def resolve_expired_trades():
    """Mark trades as NEUTRAL if they haven't resolved within timeout."""
    now = time.time()
    for trade_id, trade in list(active_trades.items()):
        if now - trade["timestamp"] > TRADE_RESOLUTION_TIMEOUT:
            trade["outcome"] = "NEUTRAL"
            trade["resolution_time"] = now
            completed_trades.append(trade)
            del active_trades[trade_id]
            log.info(f"Trade #{trade_id} marked NEUTRAL (timeout)")

# ======================================================
# ANALYSIS ENGINE (NEW - PHASE 3)
# ======================================================

def analyze_performance():
    """Analyze completed trades and generate insights."""
    if len(completed_trades) < 10:
        log.info("Not enough trades for analysis yet")
        return
    
    wins = [t for t in completed_trades if t["outcome"] == "WIN"]
    losses = [t for t in completed_trades if t["outcome"] == "LOSS"]
    neutrals = [t for t in completed_trades if t["outcome"] == "NEUTRAL"]
    
    total = len(completed_trades)
    win_rate = len(wins) / total * 100 if total > 0 else 0
    
    # Analyze by context dimensions
    insights = []
    
    # SL size analysis
    sl_categories = defaultdict(lambda: {"wins": 0, "losses": 0})
    for trade in completed_trades:
        if trade["outcome"] in ["WIN", "LOSS"]:
            cat = trade["context"]["sl_atr_ratio"]
            if trade["outcome"] == "WIN":
                sl_categories[cat]["wins"] += 1
            else:
                sl_categories[cat]["losses"] += 1
    
    for cat, stats in sl_categories.items():
        total_cat = stats["wins"] + stats["losses"]
        if total_cat >= 5:
            wr = stats["wins"] / total_cat * 100
            insights.append(f"SL {cat}: {wr:.1f}% WR ({stats['wins']}/{total_cat})")
    
    # Session analysis
    session_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
    for trade in completed_trades:
        if trade["outcome"] in ["WIN", "LOSS"]:
            sess = trade["context"]["session"]
            if trade["outcome"] == "WIN":
                session_stats[sess]["wins"] += 1
            else:
                session_stats[sess]["losses"] += 1
    
    for sess, stats in session_stats.items():
        total_sess = stats["wins"] + stats["losses"]
        if total_sess >= 3:
            wr = stats["wins"] / total_sess * 100
            insights.append(f"{sess} session: {wr:.1f}% WR ({stats['wins']}/{total_sess})")
    
    # Volume category analysis
    vol_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
    for trade in completed_trades:
        if trade["outcome"] in ["WIN", "LOSS"]:
            vol = trade["context"]["volume_category"]
            if trade["outcome"] == "WIN":
                vol_stats[vol]["wins"] += 1
            else:
                vol_stats[vol]["losses"] += 1
    
    for vol, stats in vol_stats.items():
        total_vol = stats["wins"] + stats["losses"]
        if total_vol >= 3:
            wr = stats["wins"] / total_vol * 100
            insights.append(f"Volume {vol}: {wr:.1f}% WR ({stats['wins']}/{total_vol})")
    
    # ATR = 0 warning
    atr_zero_count = sum(1 for t in completed_trades if t["context"]["atr_zero"])
    if atr_zero_count > 0:
        insights.append(f"⚠️ {atr_zero_count} trades had ATR ≈ 0")
    
    # Send report
    report = (
        f"📊 *PERFORMANCE REPORT*\n\n"
        f"Total Trades: {total}\n"
        f"Wins: {len(wins)} | Losses: {len(losses)} | Neutral: {len(neutrals)}\n"
        f"Win Rate: {win_rate:.1f}%\n\n"
        f"*Key Insights:*\n"
    )
    
    for insight in insights[:10]:  # Limit to top 10 insights
        report += f"• {insight}\n"
    
    send_telegram(report)
    log.info("Performance report sent")

# ======================================================
# SIGNAL WITH TRACKING (ENHANCED)
# ======================================================

def send_signal(symbol, direction, price, atr, df5, df15, exchange):
    global trade_counter
    
    if direction == "LONG":
        sl = price - 1.3 * atr
        tp1 = price + 2.0 * atr
        tp2 = price + 4.0 * atr
        tp3 = price + 7.0 * atr
        tp4 = price + 12.0 * atr
    else:
        sl = price + 1.3 * atr
        tp1 = price - 2.0 * atr
        tp2 = price - 4.0 * atr
        tp3 = price - 7.0 * atr
        tp4 = price - 12.0 * atr
    
    # Capture context
    context = capture_context(symbol, direction, price, sl, atr, df5, df15)
    
    # Create tracked trade
    trade = create_trade(symbol, direction, price, sl, tp1, tp2, tp3, tp4, atr, context, exchange)
    
    # Confidence assessment
    confidence = "⚠️ Low"
    warnings = []
    
    if context["sl_atr_ratio"] in ["very_tight", "tight"]:
        warnings.append("Tight SL")
    if context["atr_zero"]:
        warnings.append("ATR ≈ 0")
    if context["volume_category"] == "normal":
        warnings.append("Low volume")
    
    if len(warnings) == 0:
        confidence = "✅ High"
    elif len(warnings) == 1:
        confidence = "⚡ Medium"
    
    lv = (
        "10–20x" if ("BTC" in symbol or "ETH" in symbol)
        else "8–15x" if any(x in symbol for x in ["SOL","AVAX","LINK","BNB"])
        else "5–10x"
    )
    
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    msg = (
        f"🔥 EXPLOSIVE RR {direction} (#{trade['id']})\n\n"
        f"Pair: {symbol}\n"
        f"Entry: {round(price,6)}\n"
        f"ATR: {round(atr,6)}\n\n"
        f"SL:  {round(sl,6)}\n"
        f"TP1: {round(tp1,6)}\n"
        f"TP2: {round(tp2,6)}\n"
        f"TP3: {round(tp3,6)}\n"
        f"TP4: {round(tp4,6)}\n\n"
        f"Suggested Leverage: {lv}\n"
        f"Confidence: {confidence}\n"
    )
    
    if warnings:
        msg += f"Warnings: {', '.join(warnings)}\n"
    
    msg += (
        f"\n📊 Context:\n"
        f"• Trend: {context['trend_strength']}\n"
        f"• Volume: {context['volume_category']}\n"
        f"• Session: {context['session']}\n"
        f"• SL Size: {context['sl_atr_ratio']}\n\n"
        f"Time: {ts}"
    )
    
    send_telegram(msg)
    log.info(f"Signal sent → {symbol} {direction} (Trade #{trade['id']})")

# ======================================================
# TRADE MONITORING LOOP (NEW)
# ======================================================

def monitor_trades():
    """Monitor active trades for resolution."""
    log.info("Trade monitoring started")
    
    while True:
        try:
            # Check each active trade
            for trade_id, trade in list(active_trades.items()):
                ex = get_ex(trade["exchange"])
                if ex is None:
                    continue
                
                if check_trade_outcome(trade, ex):
                    # Trade resolved
                    completed_trades.append(trade)
                    del active_trades[trade_id]
                    
                    outcome_emoji = "✅" if trade["outcome"] == "WIN" else "❌"
                    duration = (trade["resolution_time"] - trade["timestamp"]) / 60
                    
                    msg = (
                        f"{outcome_emoji} *TRADE #{trade_id} {trade['outcome']}*\n\n"
                        f"Pair: {trade['symbol']}\n"
                        f"Direction: {trade['direction']}\n"
                        f"Hit: {trade['hit_level']}\n"
                        f"Duration: {duration:.1f} min\n"
                    )
                    send_telegram(msg)
            
            # Resolve expired trades
            resolve_expired_trades()
            
            time.sleep(10)  # Check every 10 seconds
            
        except Exception as e:
            log.error(f"Trade monitoring error: {e}")
            time.sleep(10)

# ======================================================
# ANALYSIS LOOP (NEW)
# ======================================================

def analysis_loop():
    """Periodic analysis of performance."""
    log.info("Analysis loop started")
    
    while True:
        try:
            time.sleep(ANALYSIS_INTERVAL)
            analyze_performance()
        except Exception as e:
            log.error(f"Analysis error: {e}")

# ======================================================
# MAIN SCANNER LOOP (ENHANCED)
# ======================================================

def scanner_loop():
    send_startup()
    log.info("Scanner loop started")
    
    while True:
        for ex_name in EXCHANGES:
            ex = get_ex(ex_name)
            if not ex:
                continue
            
            movers = detect_top_movers(ex)
            
            for symbol in movers:
                try:
                    df5 = get_df(ex, symbol, "5m")
                    df15 = get_df(ex, symbol, "15m")
                    
                    if df5 is None or df15 is None:
                        continue
                    
                    last = df5.iloc[-1]
                    atr = last["atr"]
                    
                    if breakout_long(df5, df15):
                        if allow(symbol, "LONG"):
                            send_signal(symbol, "LONG", last["close"], atr, df5, df15, ex_name)
                    
                    if breakout_short(df5, df15):
                        if allow(symbol, "SHORT"):
                            send_signal(symbol, "SHORT", last["close"], atr, df5, df15, ex_name)
                
                except Exception as e:
                    log.error(f"Scanner error {symbol}: {e}")
        
        time.sleep(SCAN_INTERVAL)

# ======================================================
# FLASK SERVER (ENHANCED)
# ======================================================

app = Flask(__name__)

@app.route("/")
def home():
    return "SELF-ASSESSMENT S&D BOT V2 RUNNING"

@app.route("/stats")
def stats():
    """API endpoint for statistics."""
    wins = [t for t in completed_trades if t["outcome"] == "WIN"]
    losses = [t for t in completed_trades if t["outcome"] == "LOSS"]
    neutrals = [t for t in completed_trades if t["outcome"] == "NEUTRAL"]
    
    total = len(completed_trades)
    win_rate = len(wins) / total * 100 if total > 0 else 0
    
    return jsonify({
        "total_trades": total,
        "active_trades": len(active_trades),
        "wins": len(wins),
        "losses": len(losses),
        "neutrals": len(neutrals),
        "win_rate": round(win_rate, 2),
        "active_trade_ids": list(active_trades.keys())
    })

@app.route("/trades")
def trades():
    """API endpoint to view recent trades."""
    recent = completed_trades[-20:] if len(completed_trades) > 20 else completed_trades
    
    trades_data = []
    for t in recent:
        trades_data.append({
            "id": t["id"],
            "symbol": t["symbol"],
            "direction": t["direction"],
            "outcome": t["outcome"],
            "entry": t["entry"],
            "context": t["context"]
        })
    
    return jsonify(trades_data)

if __name__ == "__main__":
    # Start all threads
    threading.Thread(target=scanner_loop, daemon=True).start()
    threading.Thread(target=monitor_trades, daemon=True).start()
    threading.Thread(target=analysis_loop, daemon=True).start()
    
    app.run(host="0.0.0.0", port=PORT)