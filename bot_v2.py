# ======================================================
# SCALPER 2.0 — INSTITUTIONAL GRADE WITH SELF-LEARNING
# High Win Rate + High R:R + Adaptive Learning System
# ======================================================

import os
import time
import ccxt
import pandas as pd
import numpy as np
import threading
from flask import Flask, jsonify
import requests
import logging
from datetime import datetime, timezone
from collections import defaultdict

# ======================================================
# LOGGING SETUP
# ======================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("SCALPER_V2")

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
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 15))
PAIR_LIMIT = int(os.getenv("PAIR_LIMIT", 80))
TOP_MOVER_COUNT = int(os.getenv("TOP_MOVER_COUNT", 15))
WINDOW = int(os.getenv("WINDOW", 1800))

EXCHANGES = ["binance", "binance_futures", "kucoin", "bybit", "okx"]

# Learning settings
TRADE_RESOLUTION_TIMEOUT = 3600
ANALYSIS_INTERVAL = 1800
MIN_TRADES_FOR_LEARNING = 20

# ======================================================
# DATA STRUCTURES
# ======================================================

recent_signals = {}
active_trades = {}
completed_trades = []
trade_counter = 0

# Enhanced learning system
learning_data = {
    "context_performance": defaultdict(lambda: {"wins": 0, "losses": 0, "total": 0, "avg_rr": 0}),
    "session_performance": defaultdict(lambda: {"wins": 0, "losses": 0}),
    "regime_performance": defaultdict(lambda: {"wins": 0, "losses": 0}),
    "filters_applied": False,
    "blacklisted_contexts": set(),
    "min_confidence_threshold": 45.0,
    "stop_hunt_count": 0,
    "fake_breakout_count": 0,
    "last_analysis_time": 0
}

# ======================================================
# TELEGRAM UTILITIES
# ======================================================

def send_telegram(text: str):
    if not BOT_TOKEN or not CHAT_IDS:
        log.error("Telegram not configured")
        return
    
    max_length = 4000
    if len(text) > max_length:
        parts = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        for part in parts:
            encoded = requests.utils.quote(part)
            for cid in CHAT_IDS:
                try:
                    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?chat_id={cid}&text={encoded}&parse_mode=Markdown"
                    requests.get(url, timeout=5)
                    time.sleep(0.5)
                except Exception as e:
                    log.error(f"Telegram error: {e}")
    else:
        encoded = requests.utils.quote(text)
        for cid in CHAT_IDS:
            try:
                url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?chat_id={cid}&text={encoded}&parse_mode=Markdown"
                requests.get(url, timeout=5)
            except Exception as e:
                log.error(f"Telegram error: {e}")

def send_startup():
    msg = (
        "🚀 *SCALPER 2.0 — ACTIVE 24/7*\n\n"
        "📊 *CORE FILTERS:*\n"
        "• Dual-timeframe regime (relaxed)\n"
        "• Trend alignment (EMA 9/20/50)\n"
        "• Displacement confirmation\n"
        "• Liquidity distance check\n"
        "• Structure-based SL (2.0-2.5× ATR)\n"
        "• Tiered profit targets (1.5R+)\n\n"
        "🧠 *LEARNING SYSTEM:*\n"
        "• Tracks stop hunts & fake breakouts\n"
        "• Learns optimal contexts\n"
        "• Auto-filters low-quality setups\n"
        "• Adapts confidence thresholds\n"
        "• Session performance tracking\n\n"
        "⚡ *SIGNAL FREQUENCY:*\n"
        "• Balanced filters for quality\n"
        "• 24/7 scanning (no session restrictions)\n"
        "• Top 15 movers per exchange\n"
        "• 15-second scan interval\n\n"
        "🎯 *TARGET METRICS:*\n"
        "• Win Rate: 55-65%\n"
        "• Avg R:R: 1.5-2.5\n\n"
        "Scanner active — hunting high-probability setups ✅"
    )
    send_telegram(msg)
    log.info(f"Startup sent → {CHAT_IDS}")

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
# ENHANCED INDICATORS
# ======================================================

def add_indicators(df):
    # EMAs
    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    
    # Volume
    df["vol_mean"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / (df["vol_mean"] + 1e-10)
    
    # ATR
    df["tr"] = df[["high", "low"]].apply(lambda x: x["high"] - x["low"], axis=1)
    df["atr"] = df["tr"].rolling(14).mean()
    df["atr_mean"] = df["atr"].rolling(14).mean()
    
    # Range metrics
    df["range"] = df["high"] - df["low"]
    df["body"] = abs(df["close"] - df["open"])
    df["body_ratio"] = df["body"] / (df["range"] + 1e-10)
    
    # EMA distance
    df["ema_dist"] = abs(df["ema9"] - df["ema20"]) / df["close"]
    
    return df

def get_df(ex, symbol, tf, retries=2):
    """Fetch OHLCV with retry logic."""
    for attempt in range(retries):
        try:
            data = ex.fetch_ohlcv(symbol, tf, limit=150)
            if not data or len(data) < 20:
                if attempt < retries - 1:
                    time.sleep(0.5)
                    continue
                return None
            
            df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
            df = add_indicators(df)
            
            # Validate indicators were calculated
            if df["atr"].isna().all() or df["ema9"].isna().all():
                return None
            
            return df
            
        except ccxt.RateLimitExceeded:
            log.warning(f"Rate limit hit for {symbol}, waiting...")
            time.sleep(2)
            continue
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(0.5)
                continue
            log.error(f"Fetch error {symbol} {tf}: {str(e)[:100]}")
            return None
    
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
        log.error(f"Exchange error ({name}): {e}")
        return None

def get_pairs(ex):
    """Get tradeable pairs with better filtering."""
    try:
        mk = ex.load_markets()
        
        # Filter for USDT pairs that are active
        pairs = []
        for symbol, market in mk.items():
            if symbol.endswith("USDT") and market.get("active", True):
                # Skip leverage tokens
                base = symbol.replace("/USDT", "")
                if not any(x in base for x in ["UP", "DOWN", "BEAR", "BULL", "HEDGE"]):
                    pairs.append(symbol)
        
        # Prioritize major pairs
        major_pairs = [p for p in pairs if any(m in p for m in ["BTC", "ETH", "BNB", "SOL", "XRP"])]
        other_pairs = [p for p in pairs if p not in major_pairs]
        
        ordered = major_pairs + other_pairs
        result = ordered[:PAIR_LIMIT]
        
        log.info(f"Loaded {len(result)} pairs from {ex.id}")
        return result
        
    except Exception as e:
        log.error(f"Error loading markets from {ex.id}: {e}")
        return []

# ======================================================
# TOP MOVERS (ENHANCED)
# ======================================================

def detect_top_movers(ex):
    """Enhanced with better error handling and diagnostics."""
    movers = []
    pairs = get_pairs(ex)
    
    if not pairs:
        log.warning(f"No pairs returned from {ex.id}")
        return []
    
    log.info(f"Scanning {len(pairs)} pairs on {ex.id}...")
    valid_count = 0
    error_count = 0
    
    for s in pairs:
        try:
            df = get_df(ex, s, "15m")
            if df is None or len(df) < 20:
                continue
            
            # More lenient data validation
            close_curr = df["close"].iloc[-1]
            close_prev = df["close"].iloc[-5]
            
            # Skip if prices are invalid
            if pd.isna(close_curr) or pd.isna(close_prev) or close_curr <= 0 or close_prev <= 0:
                continue
            
            # Calculate percentage change
            pct_change = ((close_curr - close_prev) / close_prev) * 100
            
            # Volume ratio with fallback
            vol_curr = df["volume"].iloc[-1]
            vol_mean = df["vol_mean"].iloc[-1]
            
            if pd.isna(vol_mean) or vol_mean <= 0 or pd.isna(vol_curr):
                vol_ratio = 1.0
            else:
                vol_ratio = vol_curr / vol_mean
            
            # ATR expansion with fallback
            atr_curr = df["atr"].iloc[-1]
            atr_mean = df["atr_mean"].iloc[-1]
            
            if pd.isna(atr_curr) or pd.isna(atr_mean) or atr_mean <= 0:
                atr_expansion = 1.0
            else:
                atr_expansion = atr_curr / atr_mean
            
            # Skip if any core metric is invalid
            if pd.isna(pct_change):
                continue
            
            # Simplified scoring (more lenient)
            score = (abs(pct_change) * 0.6) + (vol_ratio * 0.3) + (atr_expansion * 0.1)
            
            # More lenient threshold
            if not pd.isna(score) and score > 0.5:
                movers.append((s, score))
                valid_count += 1
        
        except Exception as e:
            error_count += 1
            if error_count <= 3:
                log.debug(f"Error processing {s}: {str(e)[:100]}")
            continue
    
    # Diagnostic logging
    log.info(f"✓ {ex.id}: {valid_count} valid movers from {len(pairs)} pairs ({error_count} errors)")
    
    if valid_count == 0:
        log.warning(f"{ex.id}: No valid movers - trying debug sample...")
        # Sample one pair for debugging
        if pairs:
            try:
                test_pair = pairs[0]
                test_df = get_df(ex, test_pair, "15m")
                if test_df is not None:
                    log.info(f"Sample {test_pair}: len={len(test_df)}, close={test_df['close'].iloc[-1]:.4f}")
                else:
                    log.warning(f"Sample {test_pair} returned None - exchange issue?")
            except Exception as e:
                log.error(f"Debug sampling failed: {e}")
        return []
    
    # Sort and return top movers
    movers_sorted = sorted(movers, key=lambda x: x[1], reverse=True)
    top_movers = [m[0] for m in movers_sorted[:TOP_MOVER_COUNT]]
    
    log.info(f"→ Top {len(top_movers)} movers: {', '.join(top_movers[:5])}...")
    return top_movers

# ======================================================
# LAYER 1 — MICRO REGIME FILTER
# ======================================================

def check_regime(df_ltf, df_htf):
    """Dual-timeframe regime check (RELAXED)."""
    if pd.isna(df_htf["atr"].iloc[-1]) or pd.isna(df_htf["atr_mean"].iloc[-1]):
        return False, "invalid_htf_atr"
    if pd.isna(df_ltf["atr"].iloc[-1]) or pd.isna(df_ltf["atr_mean"].iloc[-1]):
        return False, "invalid_ltf_atr"
    
    htf_atr_ok = df_htf["atr"].iloc[-1] >= df_htf["atr_mean"].iloc[-1] * 0.95
    ltf_atr_ok = df_ltf["atr"].iloc[-1] > df_ltf["atr_mean"].iloc[-1] * 0.9
    ltf_vol_ok = df_ltf["vol_ratio"].iloc[-1] > 0.8
    
    htf_expanding = df_htf["atr"].iloc[-1] > df_htf["atr"].iloc[-2]
    ltf_contracting = df_ltf["atr"].iloc[-1] < df_ltf["atr"].iloc[-2] * 0.95
    
    if htf_expanding and ltf_contracting:
        return False, "regime_divergence"
    
    if ltf_atr_ok and ltf_vol_ok:
        return True, "clean_expansion"
    
    return False, "low_volatility"

def categorize_regime(df_ltf, df_htf):
    """Categorize regime strength for position sizing."""
    htf_atr = df_htf["atr"].iloc[-1]
    htf_mean = df_htf["atr_mean"].iloc[-1]
    ltf_atr = df_ltf["atr"].iloc[-1]
    ltf_mean = df_ltf["atr_mean"].iloc[-1]
    
    if htf_mean > 0 and not pd.isna(htf_mean):
        htf_strength = htf_atr / htf_mean
    else:
        htf_strength = 1.0
    
    if ltf_mean > 0 and not pd.isna(ltf_mean):
        ltf_strength = ltf_atr / ltf_mean
    else:
        ltf_strength = 1.0
    
    vol_strength = df_ltf["vol_ratio"].iloc[-1]
    if pd.isna(vol_strength):
        vol_strength = 1.0
    
    avg_strength = (htf_strength + ltf_strength + vol_strength) / 3
    
    if avg_strength > 1.3:
        return "strong"
    elif avg_strength > 1.1:
        return "normal"
    else:
        return "weak"

# ======================================================
# LAYER 2 — DYNAMIC DEALING RANGE
# ======================================================

def find_last_impulse_leg(df):
    """Find the most recent impulsive move."""
    impulses = []
    
    for i in range(len(df)-10, len(df)-1):
        candle = df.iloc[i]
        
        if candle["body_ratio"] > 0.6 and candle["range"] > candle["atr"] * 1.2:
            impulses.append({
                "index": i,
                "high": candle["high"],
                "low": candle["low"],
                "direction": "bull" if candle["close"] > candle["open"] else "bear"
            })
    
    return impulses[-1] if impulses else None

def calculate_dealing_range(df, direction):
    """Calculate equilibrium based on last impulse."""
    impulse = find_last_impulse_leg(df)
    
    if not impulse:
        return None, None
    
    impulse_range = impulse["high"] - impulse["low"]
    equilibrium = impulse["low"] + (impulse_range * 0.5)
    
    discount_high = impulse["low"] + (impulse_range * 0.4)
    premium_low = impulse["high"] - (impulse_range * 0.4)
    
    if direction == "LONG":
        return impulse["low"], discount_high
    else:
        return premium_low, impulse["high"]

# ======================================================
# LAYER 3 — LIQUIDITY FILTER
# ======================================================

def find_liquidity_levels(df):
    """Find recent swing highs/lows."""
    highs = []
    lows = []
    
    for i in range(5, len(df)-5):
        if df["high"].iloc[i] > df["high"].iloc[i-1] and df["high"].iloc[i] > df["high"].iloc[i+1]:
            highs.append(df["high"].iloc[i])
        
        if df["low"].iloc[i] < df["low"].iloc[i-1] and df["low"].iloc[i] < df["low"].iloc[i+1]:
            lows.append(df["low"].iloc[i])
    
    return highs[-3:] if highs else [], lows[-3:] if lows else []

def check_liquidity_distance_relaxed(price, direction, df, atr):
    """Relaxed liquidity distance check."""
    highs, lows = find_liquidity_levels(df)
    
    min_distance = 1.0 * atr
    
    if direction == "LONG":
        if not highs:
            return True
        nearest_high = min(highs, key=lambda x: abs(x - price))
        distance = nearest_high - price
        return distance >= min_distance
    
    else:
        if not lows:
            return True
        nearest_low = min(lows, key=lambda x: abs(x - price))
        distance = price - nearest_low
        return distance >= min_distance

# ======================================================
# LAYER 4 — SESSION FILTER
# ======================================================

def get_trading_session():
    """Get current session."""
    now = datetime.now(timezone.utc)
    hour = now.hour
    minute = now.minute
    
    if hour == 8 or (hour == 9 and minute <= 30):
        return "london_open"
    elif hour >= 13 and hour < 15:
        return "ny_open"
    
    return "off_hours"

def session_momentum_ok(session):
    """Allow all sessions."""
    return True

# ======================================================
# LAYER 5 — DISPLACEMENT CONFIRMATION
# ======================================================

def check_displacement(candle, atr, vol_mean):
    """Verify clean displacement."""
    range_ok = candle["range"] >= atr * 1.2
    volume_ok = candle["volume"] >= vol_mean * 1.3
    body_ok = candle["body_ratio"] >= 0.55
    
    return range_ok and volume_ok and body_ok

# ======================================================
# LAYER 6 — EMA COMPRESSION → EXPANSION
# ======================================================

def check_ema_setup(df):
    """Detect EMA compression followed by expansion."""
    ema_dist_prev = df["ema_dist"].iloc[-3]
    ema_dist_curr = df["ema_dist"].iloc[-1]
    
    compressed = ema_dist_prev < 0.01
    expanding = ema_dist_curr > ema_dist_prev * 1.2
    
    ema9_accel = abs(df["ema9"].iloc[-1] - df["ema9"].iloc[-2]) > abs(df["ema9"].iloc[-2] - df["ema9"].iloc[-3])
    
    return compressed and expanding and ema9_accel

# ======================================================
# CORE STRATEGY LOGIC
# ======================================================

def analyze_long_setup(df5, df15):
    """Complete long setup analysis."""
    regime_ok, regime_reason = check_regime(df5, df15)
    if not regime_ok:
        return False, regime_reason
    
    trend_ok = (
        df5["ema9"].iloc[-1] > df5["ema20"].iloc[-1] > df5["ema50"].iloc[-1] and
        df15["ema9"].iloc[-1] > df15["ema20"].iloc[-1]
    )
    if not trend_ok:
        return False, "trend_misalignment"
    
    session = get_trading_session()
    
    range_low, range_high = calculate_dealing_range(df15, "LONG")
    current_price = df5["close"].iloc[-1]
    
    in_zone = True
    if range_low and range_high:
        in_zone = (range_low <= current_price <= range_high)
    
    ema_setup = check_ema_setup(df5)
    
    last_candle = df5.iloc[-1]
    atr = last_candle["atr"]
    vol_mean = df5["vol_mean"].iloc[-1]
    
    if not check_displacement(last_candle, atr, vol_mean):
        return False, "weak_displacement"
    
    if not check_liquidity_distance_relaxed(current_price, "LONG", df5, atr):
        return False, "too_close_to_liquidity"
    
    recent_high = df5["high"].iloc[-5:-1].max()
    if current_price <= recent_high * 1.0003:
        return False, "no_breakout"
    
    return True, "valid_long_setup"

def analyze_short_setup(df5, df15):
    """Complete short setup analysis."""
    regime_ok, regime_reason = check_regime(df5, df15)
    if not regime_ok:
        return False, regime_reason
    
    trend_ok = (
        df5["ema9"].iloc[-1] < df5["ema20"].iloc[-1] < df5["ema50"].iloc[-1] and
        df15["ema9"].iloc[-1] < df15["ema20"].iloc[-1]
    )
    if not trend_ok:
        return False, "trend_misalignment"
    
    session = get_trading_session()
    
    range_low, range_high = calculate_dealing_range(df15, "SHORT")
    current_price = df5["close"].iloc[-1]
    
    in_zone = True
    if range_low and range_high:
        in_zone = (range_low <= current_price <= range_high)
    
    ema_setup = check_ema_setup(df5)
    
    last_candle = df5.iloc[-1]
    atr = last_candle["atr"]
    vol_mean = df5["vol_mean"].iloc[-1]
    
    if not check_displacement(last_candle, atr, vol_mean):
        return False, "weak_displacement"
    
    if not check_liquidity_distance_relaxed(current_price, "SHORT", df5, atr):
        return False, "too_close_to_liquidity"
    
    recent_low = df5["low"].iloc[-5:-1].min()
    if current_price >= recent_low * 0.9997:
        return False, "no_breakdown"
    
    return True, "valid_short_setup"

# ======================================================
# LAYER 7 — TRADE MANAGEMENT
# ======================================================

def calculate_structure_sl(direction, entry, df, atr):
    """SL = structure + 0.5× ATR buffer."""
    if direction == "LONG":
        lows = []
        for i in range(len(df)-15, len(df)-2):
            if df["low"].iloc[i] < df["low"].iloc[i-1] and df["low"].iloc[i] < df["low"].iloc[i+1]:
                lows.append(df["low"].iloc[i])
        
        if lows:
            structure = min(lows)
            sl = structure - (0.5 * atr)
        else:
            sl = entry - (2.5 * atr)
        
        min_sl = entry - (2.0 * atr)
        return min(sl, min_sl)
    
    else:
        highs = []
        for i in range(len(df)-15, len(df)-2):
            if df["high"].iloc[i] > df["high"].iloc[i-1] and df["high"].iloc[i] > df["high"].iloc[i+1]:
                highs.append(df["high"].iloc[i])
        
        if highs:
            structure = max(highs)
            sl = structure + (0.5 * atr)
        else:
            sl = entry + (2.5 * atr)
        
        max_sl = entry + (2.0 * atr)
        return max(sl, max_sl)

def calculate_targets(direction, entry, sl, df):
    """Calculate tiered profit targets."""
    risk = abs(entry - sl)
    
    highs, lows = find_liquidity_levels(df)
    
    if direction == "LONG":
        tp1 = entry + (risk * 1.5)
        
        if highs:
            potential_tp2 = min([h for h in highs if h > entry], default=entry + (risk * 3))
            tp2 = max(potential_tp2, tp1 + risk)
        else:
            tp2 = entry + (risk * 3)
        
        tp3 = entry + (risk * 5)
    
    else:
        tp1 = entry - (risk * 1.5)
        
        if lows:
            potential_tp2 = max([l for l in lows if l < entry], default=entry - (risk * 3))
            tp2 = min(potential_tp2, tp1 - risk)
        else:
            tp2 = entry - (risk * 3)
        
        tp3 = entry - (risk * 5)
    
    return tp1, tp2, tp3

# ======================================================
# CONTEXT & LEARNING
# ======================================================

def capture_context(symbol, direction, entry, sl, atr, df5, df15):
    """Capture context for learning."""
    regime = categorize_regime(df5, df15)
    session = get_trading_session()
    
    sl_distance = abs(entry - sl)
    
    if atr > 0 and not pd.isna(atr):
        sl_atr_ratio = sl_distance / atr
    else:
        sl_atr_ratio = 2.0
    
    if sl_atr_ratio < 1.5:
        sl_category = "tight"
    elif sl_atr_ratio < 2.5:
        sl_category = "medium"
    else:
        sl_category = "wide"
    
    ema9 = df5["ema9"].iloc[-1]
    ema50 = df5["ema50"].iloc[-1]
    close = df5["close"].iloc[-1]
    
    if close > 0 and not pd.isna(ema9) and not pd.isna(ema50):
        ema_sep_5 = abs(ema9 - ema50) / close
    else:
        ema_sep_5 = 0.01
    
    trend_strength = "strong" if ema_sep_5 > 0.015 else "medium" if ema_sep_5 > 0.008 else "weak"
    
    vol_ratio = df5["vol_ratio"].iloc[-1]
    if pd.isna(vol_ratio):
        vol_ratio = 1.0
    
    atr_curr = df5["atr"].iloc[-1]
    atr_mean = df5["atr_mean"].iloc[-1]
    if atr_mean > 0 and not pd.isna(atr_mean) and not pd.isna(atr_curr):
        atr_expansion = atr_curr / atr_mean
    else:
        atr_expansion = 1.0
    
    context = {
        "regime": regime,
        "session": session,
        "sl_category": sl_category,
        "sl_atr_multiple": round(sl_atr_ratio, 2),
        "trend_strength": trend_strength,
        "volume_ratio": round(vol_ratio, 2),
        "atr_expansion": round(atr_expansion, 2),
        "pair_type": "major" if any(x in symbol for x in ["BTC", "ETH"]) else "alt"
    }
    
    return context

def context_to_key(context):
    """Convert context to learning key."""
    return f"{context['regime']}_{context['session']}_{context['sl_category']}_{context['trend_strength']}"

def calculate_confidence(context):
    """Calculate setup confidence based on learned data."""
    
    if len(completed_trades) < MIN_TRADES_FOR_LEARNING:
        confidence = 50.0
        
        if context["regime"] == "strong":
            confidence += 15
        if context["session"] in ["london_open", "ny_open"]:
            confidence += 10
        if context["sl_category"] in ["medium", "wide"]:
            confidence += 10
        if context["trend_strength"] == "strong":
            confidence += 10
        if context["volume_ratio"] > 1.5:
            confidence += 5
        
        return max(30, min(85, confidence))
    
    context_key = context_to_key(context)
    perf = learning_data["context_performance"][context_key]
    
    if perf["total"] >= 5:
        win_rate = (perf["wins"] / perf["total"]) * 100
        return win_rate
    
    base_confidence = 50.0
    
    session_perf = learning_data["session_performance"][context["session"]]
    if session_perf["wins"] + session_perf["losses"] >= 3:
        session_wr = (session_perf["wins"] / (session_perf["wins"] + session_perf["losses"])) * 100
        base_confidence = (base_confidence + session_wr) / 2
    
    regime_perf = learning_data["regime_performance"][context["regime"]]
    if regime_perf["wins"] + regime_perf["losses"] >= 3:
        regime_wr = (regime_perf["wins"] / (regime_perf["wins"] + regime_perf["losses"])) * 100
        base_confidence = (base_confidence + regime_wr) / 2
    
    return max(20, min(90, base_confidence))

def should_take_trade(context, confidence):
    """Decide if trade passes learning filters."""
    
    if len(completed_trades) < MIN_TRADES_FOR_LEARNING:
        return True, "learning_phase"
    
    context_key = context_to_key(context)
    if context_key in learning_data["blacklisted_contexts"]:
        return False, "blacklisted_context"
    
    threshold = learning_data["min_confidence_threshold"]
    if confidence < threshold:
        return False, f"low_confidence_{confidence:.0f}%"
    
    if learning_data["stop_hunt_count"] > 5 and context["sl_category"] == "tight":
        return False, "stop_hunt_prone"
    
    return True, "passed_filters"

# ======================================================
# TRADE TRACKING
# ======================================================

def create_trade(symbol, direction, entry, sl, tp1, tp2, tp3, atr, context, confidence, exchange):
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
        "atr": atr,
        "exchange": exchange,
        "context": context,
        "confidence": confidence,
        "timestamp": time.time(),
        "outcome": "PENDING",
        "resolution_time": None,
        "hit_level": None,
        "high_reached": entry,
        "low_reached": entry,
        "reason": None
    }
    
    active_trades[trade_counter] = trade
    log.info(f"Trade #{trade_counter}: {symbol} {direction} @ {entry}")
    
    return trade

def analyze_trade_outcome(trade):
    """Deep analysis of WHY trade won or lost."""
    reasons = []
    
    ctx = trade["context"]
    entry = trade["entry"]
    sl = trade["sl"]
    
    if trade["outcome"] == "WIN":
        if ctx["regime"] == "strong":
            reasons.append("✅ Strong regime provided momentum")
        
        if ctx["session"] in ["london_open", "ny_open"]:
            reasons.append("✅ Premium session window")
        
        if ctx["sl_category"] in ["medium", "wide"]:
            reasons.append("✅ Proper SL sizing avoided noise")
        
        if ctx["volume_ratio"] > 1.5:
            reasons.append("✅ Strong volume confirmed move")
        
        if ctx["trend_strength"] == "strong":
            reasons.append("✅ Clear directional bias")
        
        if not reasons:
            reasons.append("✅ Clean execution with follow-through")
    
    else:
        if trade["direction"] == "LONG":
            price_after_sl = trade["high_reached"]
            move_after = ((price_after_sl - sl) / sl) * 100
            
            if move_after > 0.3:
                reasons.append(f"🚨 STOP HUNT: Price rose {move_after:.2f}% after SL hit")
                learning_data["stop_hunt_count"] += 1
        else:
            price_after_sl = trade["low_reached"]
            move_after = ((sl - price_after_sl) / sl) * 100
            
            if move_after > 0.3:
                reasons.append(f"🚨 STOP HUNT: Price fell {move_after:.2f}% after SL hit")
                learning_data["stop_hunt_count"] += 1
        
        if ctx["sl_category"] == "tight":
            reasons.append("⚠️ SL too tight - vulnerable to wicks")
        
        if ctx["volume_ratio"] < 1.3:
            reasons.append("⚠️ Weak volume - insufficient momentum")
        
        if ctx["regime"] == "weak":
            reasons.append("⚠️ Weak regime - low conviction move")
        
        if ctx["session"] == "off_hours":
            reasons.append("⚠️ Off-hours trade - low liquidity")
        
        if ctx["trend_strength"] == "weak":
            reasons.append("⚠️ Weak trend - prone to reversals")
        
        if "STOP HUNT" not in " ".join(reasons):
            reasons.append("❌ False breakout - setup invalidated")
            learning_data["fake_breakout_count"] += 1
    
    return " | ".join(reasons[:4])

def check_trade_outcome(trade, ex):
    """Monitor trade for resolution."""
    try:
        symbol = trade["symbol"]
        df = get_df(ex, symbol, "5m")
        
        if df is None or len(df) == 0:
            return False
        
        high = df["high"].iloc[-1]
        low = df["low"].iloc[-1]
        
        trade["high_reached"] = max(trade["high_reached"], high)
        trade["low_reached"] = min(trade["low_reached"], low)
        
        if trade["direction"] == "LONG":
            if high >= trade["tp1"]:
                trade["outcome"] = "WIN"
                trade["hit_level"] = "TP1"
                trade["resolution_time"] = time.time()
                trade["reason"] = analyze_trade_outcome(trade)
                return True
            elif low <= trade["sl"]:
                trade["outcome"] = "LOSS"
                trade["hit_level"] = "SL"
                trade["resolution_time"] = time.time()
                trade["reason"] = analyze_trade_outcome(trade)
                return True
        
        else:
            if low <= trade["tp1"]:
                trade["outcome"] = "WIN"
                trade["hit_level"] = "TP1"
                trade["resolution_time"] = time.time()
                trade["reason"] = analyze_trade_outcome(trade)
                return True
            elif high >= trade["sl"]:
                trade["outcome"] = "LOSS"
                trade["hit_level"] = "SL"
                trade["resolution_time"] = time.time()
                trade["reason"] = analyze_trade_outcome(trade)
                return True
        
        return False
        
    except Exception as e:
        log.error(f"Error checking trade: {e}")
        return False

def resolve_expired_trades():
    """Handle trades that timeout."""
    now = time.time()
    for trade_id, trade in list(active_trades.items()):
        if now - trade["timestamp"] > TRADE_RESOLUTION_TIMEOUT:
            trade["outcome"] = "NEUTRAL"
            trade["resolution_time"] = now
            trade["reason"] = "Timeout - no clear outcome (60min)"
            completed_trades.append(trade)
            del active_trades[trade_id]
            log.info(f"Trade #{trade_id} timed out")

# ======================================================
# LEARNING SYSTEM
# ======================================================

def update_learning_data(trade):
    """Update learning system with completed trade."""
    
    ctx = trade["context"]
    context_key = context_to_key(ctx)
    
    perf = learning_data["context_performance"][context_key]
    
    if trade["outcome"] == "WIN":
        perf["wins"] += 1
        perf["total"] += 1
        
        risk = abs(trade["entry"] - trade["sl"])
        reward = abs(trade["entry"] - trade["tp1"])
        rr = reward / risk if risk > 0 else 0
        
        prev_avg = perf["avg_rr"]
        perf["avg_rr"] = ((prev_avg * (perf["total"] - 1)) + rr) / perf["total"]
    
    elif trade["outcome"] == "LOSS":
        perf["losses"] += 1
        perf["total"] += 1
    
    session_perf = learning_data["session_performance"][ctx["session"]]
    if trade["outcome"] == "WIN":
        session_perf["wins"] += 1
    elif trade["outcome"] == "LOSS":
        session_perf["losses"] += 1
    
    regime_perf = learning_data["regime_performance"][ctx["regime"]]
    if trade["outcome"] == "WIN":
        regime_perf["wins"] += 1
    elif trade["outcome"] == "LOSS":
        regime_perf["losses"] += 1
    
    if perf["total"] >= 10:
        win_rate = (perf["wins"] / perf["total"]) * 100
        if win_rate < 25:
            learning_data["blacklisted_contexts"].add(context_key)
            log.warning(f"🚫 Blacklisted: {context_key} (WR: {win_rate:.1f}%)")
    
    if len(completed_trades) >= MIN_TRADES_FOR_LEARNING:
        recent_trades = completed_trades[-20:]
        recent_wins = sum(1 for t in recent_trades if t["outcome"] == "WIN")
        recent_wr = (recent_wins / len(recent_trades)) * 100
        
        if recent_wr < 50:
            learning_data["min_confidence_threshold"] = min(65, learning_data["min_confidence_threshold"] + 2)
        elif recent_wr > 65:
            learning_data["min_confidence_threshold"] = max(40, learning_data["min_confidence_threshold"] - 1)

def analyze_performance():
    """Generate detailed performance report."""
    
    if len(completed_trades) < 5:
        log.info("Need more trades for analysis")
        return
    
    wins = [t for t in completed_trades if t["outcome"] == "WIN"]
    losses = [t for t in completed_trades if t["outcome"] == "LOSS"]
    neutrals = [t for t in completed_trades if t["outcome"] == "NEUTRAL"]
    
    total = len(completed_trades)
    win_rate = (len(wins) / total * 100) if total > 0 else 0
    
    total_rr = 0
    for w in wins:
        risk = abs(w["entry"] - w["sl"])
        reward = abs(w["entry"] - w["tp1"])
        if risk > 0:
            total_rr += (reward / risk)
    
    avg_rr = (total_rr / len(wins)) if wins else 0
    
    report = [
        "📊 *PERFORMANCE ANALYSIS*\n",
        f"Total Trades: {total}",
        f"Wins: {len(wins)} | Losses: {len(losses)} | Neutral: {len(neutrals)}",
        f"Win Rate: *{win_rate:.1f}%*",
        f"Avg R:R: *{avg_rr:.2f}*\n"
    ]
    
    if learning_data["stop_hunt_count"] > 0:
        report.append(f"🚨 Stop Hunts Detected: {learning_data['stop_hunt_count']}")
    
    if learning_data["fake_breakout_count"] > 0:
        report.append(f"⚠️ Fake Breakouts: {learning_data['fake_breakout_count']}\n")
    
    report.append("*Session Performance:*")
    for session, perf in learning_data["session_performance"].items():
        total_s = perf["wins"] + perf["losses"]
        if total_s >= 3:
            wr_s = (perf["wins"] / total_s) * 100
            report.append(f"• {session}: {wr_s:.0f}% ({perf['wins']}/{total_s})")
    
    report.append("\n*Regime Performance:*")
    for regime, perf in learning_data["regime_performance"].items():
        total_r = perf["wins"] + perf["losses"]
        if total_r >= 3:
            wr_r = (perf["wins"] / total_r) * 100
            report.append(f"• {regime}: {wr_r:.0f}% ({perf['wins']}/{total_r})")
    
    report.append("\n*Top Performing Setups:*")
    sorted_contexts = sorted(
        learning_data["context_performance"].items(),
        key=lambda x: x[1]["wins"] if x[1]["total"] >= 5 else 0,
        reverse=True
    )[:3]
    
    for ctx_key, perf in sorted_contexts:
        if perf["total"] >= 5:
            wr_c = (perf["wins"] / perf["total"]) * 100
            report.append(f"• {ctx_key}: {wr_c:.0f}% ({perf['wins']}/{perf['total']})")
    
    if learning_data["blacklisted_contexts"]:
        report.append(f"\n🚫 Blacklisted Contexts: {len(learning_data['blacklisted_contexts'])}")
    
    report.append(f"\n⚙️ Current Min Confidence: {learning_data['min_confidence_threshold']:.0f}%")
    
    send_telegram("\n".join(report))
    log.info("Performance analysis sent")

# ======================================================
# SIGNAL GENERATION
# ======================================================

def send_signal(symbol, direction, entry, sl, tp1, tp2, tp3, atr, context, confidence, exchange, df5):
    """Send enhanced signal with learning insights."""
    
    trade = create_trade(symbol, direction, entry, sl, tp1, tp2, tp3, atr, context, confidence, exchange)
    
    risk = abs(entry - sl)
    reward1 = abs(entry - tp1)
    rr1 = reward1 / risk if risk > 0 else 0
    
    if context["regime"] == "strong":
        size_mult = "1.25-1.5x"
    elif context["regime"] == "normal":
        size_mult = "1.0x"
    else:
        size_mult = "0.5x"
    
    if confidence >= 70:
        conf_emoji = "🟢"
    elif confidence >= 55:
        conf_emoji = "🟡"
    else:
        conf_emoji = "🟠"
    
    msg = (
        f"🎯 *SCALPER 2.0 SIGNAL* #{trade['id']}\n\n"
        f"*Pair:* {symbol}\n"
        f"*Direction:* {direction}\n"
        f"*Entry:* {entry:.6f}\n"
        f"*SL:* {sl:.6f}\n\n"
        f"*Targets:*\n"
        f"TP1: {tp1:.6f} ({rr1:.1f}R)\n"
        f"TP2: {tp2:.6f}\n"
        f"TP3: {tp3:.6f}\n\n"
        f"*Risk Management:*\n"
        f"• SL Distance: {context['sl_atr_multiple']:.1f}× ATR\n"
        f"• Position Size: {size_mult}\n\n"
        f"*Setup Quality:*\n"
        f"{conf_emoji} Confidence: {confidence:.0f}%\n"
        f"• Regime: {context['regime']}\n"
        f"• Session: {context['session']}\n"
        f"• Trend: {context['trend_strength']}\n"
        f"• Volume: {context['volume_ratio']}x\n"
        f"• ATR Expansion: {context['atr_expansion']}x\n\n"
    )
    
    if len(completed_trades) >= MIN_TRADES_FOR_LEARNING:
        context_key = context_to_key(context)
        perf = learning_data["context_performance"][context_key]
        
        if perf["total"] >= 3:
            ctx_wr = (perf["wins"] / perf["total"]) * 100
            msg += f"📚 *Historical Data:*\n"
            msg += f"Similar setups: {ctx_wr:.0f}% WR ({perf['wins']}/{perf['total']})\n"
            if perf["avg_rr"] > 0:
                msg += f"Avg R:R: {perf['avg_rr']:.1f}\n\n"
    
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    msg += f"⏰ {ts}"
    
    send_telegram(msg)
    log.info(f"Signal sent: {symbol} {direction} #{trade['id']}")

# ======================================================
# TRADE MONITORING
# ======================================================

def monitor_trades():
    """Monitor active trades for resolution."""
    log.info("Trade monitoring started")
    
    while True:
        try:
            for trade_id, trade in list(active_trades.items()):
                ex = get_ex(trade["exchange"])
                if ex is None:
                    continue
                
                if check_trade_outcome(trade, ex):
                    completed_trades.append(trade)
                    del active_trades[trade_id]
                    
                    update_learning_data(trade)
                    
                    duration = (trade["resolution_time"] - trade["timestamp"]) / 60
                    
                    if trade["outcome"] == "WIN":
                        risk = abs(trade["entry"] - trade["sl"])
                        reward = abs(trade["entry"] - trade["tp1"])
                        rr_achieved = reward / risk if risk > 0 else 0
                        
                        msg = (
                            f"✅ *TRADE #{trade_id} WIN*\n\n"
                            f"Pair: {trade['symbol']}\n"
                            f"Direction: {trade['direction']}\n"
                            f"Entry: {trade['entry']:.6f}\n"
                            f"Exit: {trade['tp1']:.6f}\n"
                            f"R:R Achieved: {rr_achieved:.1f}R\n"
                            f"Duration: {duration:.1f} min\n\n"
                            f"*Analysis:*\n{trade['reason']}"
                        )
                    else:
                        msg = (
                            f"❌ *TRADE #{trade_id} LOSS*\n\n"
                            f"Pair: {trade['symbol']}\n"
                            f"Direction: {trade['direction']}\n"
                            f"Entry: {trade['entry']:.6f}\n"
                            f"SL Hit: {trade['sl']:.6f}\n"
                            f"Duration: {duration:.1f} min\n\n"
                            f"*Analysis:*\n{trade['reason']}"
                        )
                    
                    send_telegram(msg)
            
            resolve_expired_trades()
            time.sleep(10)
            
        except Exception as e:
            log.error(f"Monitoring error: {e}")
            time.sleep(10)

# ======================================================
# ANALYSIS LOOP
# ======================================================

def analysis_loop():
    """Periodic performance analysis."""
    log.info("Analysis loop started")
    
    while True:
        try:
            time.sleep(ANALYSIS_INTERVAL)
            
            if len(completed_trades) >= 5:
                analyze_performance()
            
        except Exception as e:
            log.error(f"Analysis error: {e}")

# ======================================================
# MAIN SCANNER
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
                    if len(df5) < 50 or len(df15) < 50:
                        continue
                    
                    long_valid, long_reason = analyze_long_setup(df5, df15)
                    
                    if long_valid and allow(symbol, "LONG"):
                        entry = df5["close"].iloc[-1]
                        atr = df5["atr"].iloc[-1]
                        
                        sl = calculate_structure_sl("LONG", entry, df5, atr)
                        tp1, tp2, tp3 = calculate_targets("LONG", entry, sl, df15)
                        
                        context = capture_context(symbol, "LONG", entry, sl, atr, df5, df15)
                        confidence = calculate_confidence(context)
                        
                        take_trade, filter_reason = should_take_trade(context, confidence)
                        
                        if take_trade:
                            send_signal(symbol, "LONG", entry, sl, tp1, tp2, tp3, atr, context, confidence, ex_name, df5)
                        else:
                            log.info(f"Filtered {symbol} LONG: {filter_reason}")
                    
                    short_valid, short_reason = analyze_short_setup(df5, df15)
                    
                    if short_valid and allow(symbol, "SHORT"):
                        entry = df5["close"].iloc[-1]
                        atr = df5["atr"].iloc[-1]
                        
                        sl = calculate_structure_sl("SHORT", entry, df5, atr)
                        tp1, tp2, tp3 = calculate_targets("SHORT", entry, sl, df15)
                        
                        context = capture_context(symbol, "SHORT", entry, sl, atr, df5, df15)
                        confidence = calculate_confidence(context)
                        
                        take_trade, filter_reason = should_take_trade(context, confidence)
                        
                        if take_trade:
                            send_signal(symbol, "SHORT", entry, sl, tp1, tp2, tp3, atr, context, confidence, ex_name, df5)
                        else:
                            log.info(f"Filtered {symbol} SHORT: {filter_reason}")
                
                except Exception as e:
                    log.error(f"Scanner error {symbol}: {e}")
        
        time.sleep(SCAN_INTERVAL)

# ======================================================
# FLASK SERVER
# ======================================================

app = Flask(__name__)

@app.route("/")
def home():
    return "SCALPER 2.0 — INSTITUTIONAL GRADE RUNNING ✅"

@app.route("/stats")
def stats():
    wins = [t for t in completed_trades if t["outcome"] == "WIN"]
    losses = [t for t in completed_trades if t["outcome"] == "LOSS"]
    neutrals = [t for t in completed_trades if t["outcome"] == "NEUTRAL"]
    
    total = len(completed_trades)
    win_rate = (len(wins) / total * 100) if total > 0 else 0
    
    total_rr = 0
    for w in wins:
        risk = abs(w["entry"] - w["sl"])
        reward = abs(w["entry"] - w["tp1"])
        if risk > 0:
            total_rr += (reward / risk)
    avg_rr = (total_rr / len(wins)) if wins else 0
    
    return jsonify({
        "total_trades": total,
        "active_trades": len(active_trades),
        "wins": len(wins),
        "losses": len(losses),
        "neutrals": len(neutrals),
        "win_rate": round(win_rate, 2),
        "avg_rr": round(avg_rr, 2),
        "stop_hunts": learning_data["stop_hunt_count"],
        "fake_breakouts": learning_data["fake_breakout_count"],
        "blacklisted_contexts": len(learning_data["blacklisted_contexts"]),
        "min_confidence": learning_data["min_confidence_threshold"]
    })

@app.route("/trades")
def trades():
    recent = completed_trades[-30:] if len(completed_trades) > 30 else completed_trades
    
    trades_data = []
    for t in recent:
        trades_data.append({
            "id": t["id"],
            "symbol": t["symbol"],
            "direction": t["direction"],
            "outcome": t["outcome"],
            "entry": round(t["entry"], 6),
            "sl": round(t["sl"], 6),
            "confidence": t["confidence"],
            "reason": t["reason"],
            "context": t["context"]
        })
    
    return jsonify(trades_data)

@app.route("/learning")
def learning():
    """View learning data."""
    return jsonify({
        "total_completed": len(completed_trades),
        "filters_active": len(completed_trades) >= MIN_TRADES_FOR_LEARNING,
        "min_confidence_threshold": learning_data["min_confidence_threshold"],
        "blacklisted_contexts": list(learning_data["blacklisted_contexts"]),
        "stop_hunt_count": learning_data["stop_hunt_count"],
        "fake_breakout_count": learning_data["fake_breakout_count"],
        "session_performance": dict(learning_data["session_performance"]),
        "regime_performance": dict(learning_data["regime_performance"])
    })

if __name__ == "__main__":
    threading.Thread(target=scanner_loop, daemon=True).start()
    threading.Thread(target=monitor_trades, daemon=True).start()
    threading.Thread(target=analysis_loop, daemon=True).start()
    
    app.run(host="0.0.0.0", port=PORT)
