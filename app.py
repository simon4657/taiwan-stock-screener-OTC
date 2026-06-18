#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
台股主力資金進入篩選器 - 上櫃市場版本
使用Pine Script技術分析邏輯，專門針對台灣上櫃市場股票進行主力資金進場信號篩選
"""

from flask import Flask, render_template, jsonify, request
import requests
import json
import math
from datetime import datetime, timedelta, timezone
import pytz
import logging
import traceback
from typing import Dict, List, Optional, Tuple, Any
import time
import urllib3
import threading

# 抑制SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
DEPLOY_VERSION = "otc-realtime-breakout-2026-06-18"

# 全域變數
stocks_data = {}
last_update_time = None
data_date = None
issued_shares_cache = None
issued_shares_cache_time = None
realtime_jobs = {}
realtime_jobs_lock = threading.Lock()

# 台灣時區
TW_TZ = pytz.timezone('Asia/Taipei')

def get_taiwan_time():
    """獲取台灣時間"""
    return datetime.now(TW_TZ)

def convert_roc_date_to_ad(roc_date_str):
    """將民國年日期轉換為西元年日期"""
    try:
        if not roc_date_str or len(roc_date_str) != 7:
            return None
        
        roc_year = int(roc_date_str[:3])
        month = int(roc_date_str[3:5])
        day = int(roc_date_str[5:7])
        
        ad_year = roc_year + 1911
        return f"{ad_year:04d}-{month:02d}-{day:02d}"
    except:
        return None

def convert_ad_date_to_roc(ad_date_str):
    """將西元年日期轉換為民國年日期"""
    try:
        if isinstance(ad_date_str, str):
            if '-' in ad_date_str:
                year, month, day = ad_date_str.split('-')
            else:
                year = ad_date_str[:4]
                month = ad_date_str[4:6]
                day = ad_date_str[6:8]
        else:
            return None
        
        roc_year = int(year) - 1911
        return f"{roc_year:03d}{int(month):02d}{int(day):02d}"
    except:
        return None

def fetch_otc_stock_data():
    """獲取上櫃股票資料"""
    try:
        logger.info("開始獲取上櫃股票資料...")
        
        # 台灣櫃買中心API
        url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # 修正SSL證書驗證問題
        response = requests.get(url, headers=headers, timeout=30, verify=False)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"成功獲取上櫃股票資料，共 {len(data)} 筆")
        
        return data
        
    except Exception as e:
        logger.error(f"獲取上櫃股票資料失敗: {str(e)}")
        return []

def process_otc_stock_data(raw_data):
    """處理上櫃股票資料"""
    processed_stocks = {}
    current_date = None
    
    try:
        for item in raw_data:
            # 獲取股票基本資訊
            stock_code = item.get('SecuritiesCompanyCode', '').strip()
            stock_name = item.get('CompanyName', '').strip()
            date_str = item.get('Date', '').strip()
            
            # 設定資料日期
            if not current_date and date_str:
                current_date = date_str
            
            # 篩選上櫃股票（排除ETF、債券等）
            if not is_valid_otc_stock(stock_code, stock_name):
                continue
            
            # 處理價格資料
            try:
                close_price = float(item.get('Close', '0').replace(',', ''))
                open_price = float(item.get('Open', '0').replace(',', ''))
                high_price = float(item.get('High', '0').replace(',', ''))
                low_price = float(item.get('Low', '0').replace(',', ''))
                volume = int(item.get('TradingShares', '0').replace(',', ''))
                
                if close_price <= 0:
                    continue
                
                # 處理漲跌幅計算
                change_str = item.get('Change', '0.00').strip()
                try:
                    if change_str.startswith('-'):
                        change = -float(change_str[1:])
                    else:
                        change = float(change_str) if change_str else 0
                    
                    # 計算漲跌幅百分比
                    change_percent = (change / (close_price - change)) * 100 if (close_price - change) != 0 else 0
                except (ValueError, TypeError):
                    change = 0
                    change_percent = 0
                
                processed_stocks[stock_code] = {
                    'code': stock_code,
                    'name': stock_name,
                    'close': close_price,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'volume': volume,
                    'date': date_str,
                    'change': change,
                    'change_percent': change_percent,
                    'transaction_amount': item.get('TransactionAmount', '0'),
                    'market': 'OTC'  # 標記為上櫃市場
                }
                
            except (ValueError, TypeError) as e:
                logger.warning(f"處理股票 {stock_code} 資料時發生錯誤: {str(e)}")
                continue
        
        logger.info(f"成功處理 {len(processed_stocks)} 支上櫃股票資料")
        return processed_stocks, current_date
        
    except Exception as e:
        logger.error(f"處理上櫃股票資料時發生錯誤: {str(e)}")
        return {}, None

def is_valid_otc_stock(stock_code, stock_name):
    """判斷是否為有效的上櫃一般股票"""
    if not stock_code or not stock_name:
        return False
    
    # 檢查股票代碼格式
    if not stock_code.isdigit() or len(stock_code) < 4:
        return False
    
    # 上櫃股票代碼範圍（一般為1000-9999）
    try:
        code_num = int(stock_code)
        if not (1000 <= code_num <= 9999):
            return False
    except ValueError:
        return False
    
    # 排除特殊股票類型
    exclude_suffixes = ['B', 'K', 'L', 'R', 'F']  # ETF、債券等
    if any(stock_code.endswith(suffix) for suffix in exclude_suffixes):
        return False
    
    # 排除特殊名稱
    exclude_keywords = ['ETF', 'ETN', '權證', '特別股', '存託憑證', '債券', 'REITs']
    if any(keyword in stock_name for keyword in exclude_keywords):
        return False
    
    return True

def clean_data_for_json(data):
    """清理數據以確保JSON序列化成功"""
    import math
    
    if isinstance(data, dict):
        return {k: clean_data_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return 0
        return data
    else:
        return data

def parse_market_number(value):
    """解析市場資料中的數字欄位。"""
    if value is None:
        return 0
    text = str(value).strip().replace(',', '').replace(' ', '')
    if text in ('', '-', '--', 'N/A', 'nan'):
        return 0
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0

def calculate_weighted_simple_average(src_values, length, weight):
    """完全按照Pine Script邏輯實現的加權移動平均"""
    if not src_values or length <= 0:
        return 0
    
    if len(src_values) == 1:
        return src_values[0]
    
    # Pine Script狀態變量
    sum_float = 0.0
    output = None
    
    # 逐步計算，維護Pine Script的狀態邏輯
    for i, src in enumerate(src_values):
        # Pine Script邏輯：sum_float := nz(sum_float[1]) - nz(src[length]) + src
        if i >= length:
            # 移除length期前的值，加入當前值
            sum_float = sum_float - src_values[i - length] + src
        else:
            # 累加當前值
            sum_float += src
        
        # 計算移動平均
        if i >= length - 1:
            moving_average = sum_float / length
        else:
            moving_average = None  # Pine Script中會是na
        
        # Pine Script邏輯：output := na(output[1]) ? moving_average : (src * weight + output[1] * (length - weight)) / length
        if output is None:
            # 第一次計算或moving_average為None時
            output = moving_average if moving_average is not None else src
        else:
            if moving_average is not None:
                # 標準的加權計算
                output = (src * weight + output * (length - weight)) / length
            else:
                # 如果moving_average為None，保持原值
                output = (src * weight + output * (length - weight)) / length
    
    return output if output is not None else (src_values[-1] if src_values else 0)

def calculate_pine_script_indicators(ohlc_data):
    """完全按照Pine Script邏輯計算技術指標"""
    if len(ohlc_data) < 34:  # 需要足夠的歷史數據
        return None
    
    # 提取OHLC數據
    closes = [d['close'] for d in ohlc_data]
    highs = [d['high'] for d in ohlc_data]
    lows = [d['low'] for d in ohlc_data]
    opens = [d['open'] for d in ohlc_data]
    
    # 計算典型價格 (2 * close + high + low + open) / 5
    typical_prices = [(2 * c + h + l + o) / 5 for c, h, l, o in zip(closes, highs, lows, opens)]
    
    # 計算資金流向趨勢（完全按照Pine Script公式）
    fund_flow_values = []
    
    for i in range(len(closes)):
        # 計算27期最高最低價
        start_idx = max(0, i - 26)
        lowest_27 = min(lows[start_idx:i+1])
        highest_27 = max(highs[start_idx:i+1])
        
        if highest_27 != lowest_27:
            # 計算相對位置
            relative_pos = (closes[i] - lowest_27) / (highest_27 - lowest_27) * 100
            
            # 收集足夠的相對位置數據用於加權平均
            relative_positions = []
            for j in range(max(0, i - 4), i + 1):
                start_j = max(0, j - 26)
                low_27_j = min(lows[start_j:j+1])
                high_27_j = max(highs[start_j:j+1])
                if high_27_j != low_27_j:
                    rel_pos_j = (closes[j] - low_27_j) / (high_27_j - low_27_j) * 100
                else:
                    rel_pos_j = 50
                relative_positions.append(rel_pos_j)
            
            # 第一層加權簡單平均（5期，權重1）
            wsa1 = calculate_weighted_simple_average(relative_positions, min(5, len(relative_positions)), 1)
            
            # 第二層加權簡單平均（3期，權重1）
            if i >= 2:
                # 收集前面的wsa1值
                wsa1_values = []
                for k in range(max(0, i - 2), i + 1):
                    # 重新計算每個時點的wsa1
                    rel_pos_k = []
                    for j in range(max(0, k - 4), k + 1):
                        start_j = max(0, j - 26)
                        low_27_j = min(lows[start_j:j+1])
                        high_27_j = max(highs[start_j:j+1])
                        if high_27_j != low_27_j:
                            rel_pos_j = (closes[j] - low_27_j) / (high_27_j - low_27_j) * 100
                        else:
                            rel_pos_j = 50
                        rel_pos_k.append(rel_pos_j)
                    
                    wsa1_k = calculate_weighted_simple_average(rel_pos_k, min(5, len(rel_pos_k)), 1)
                    wsa1_values.append(wsa1_k)
                
                wsa2 = calculate_weighted_simple_average(wsa1_values, min(3, len(wsa1_values)), 1)
            else:
                wsa2 = wsa1
            
            # 最終公式：(3 * wsa1 - 2 * wsa2 - 50) * 1.032 + 50
            fund_flow = (3 * wsa1 - 2 * wsa2 - 50) * 1.032 + 50
        else:
            fund_flow = 50
        
        fund_flow_values.append(max(0, min(100, fund_flow)))
    
    # 計算多空線（13期EMA）
    # 先計算標準化的典型價格
    bull_bear_values = []
    for i in range(len(typical_prices)):
        # 計算34期最高最低價
        start_idx = max(0, i - 33)
        lowest_34 = min(lows[start_idx:i+1])
        highest_34 = max(highs[start_idx:i+1])
        
        if highest_34 != lowest_34:
            normalized_price = (typical_prices[i] - lowest_34) / (highest_34 - lowest_34) * 100
        else:
            normalized_price = 50
        bull_bear_values.append(max(0, min(100, normalized_price)))
    
    # 計算13期EMA
    bull_bear_line_values = []
    for i in range(len(bull_bear_values)):
        if i < 13:
            ema_value = sum(bull_bear_values[:i+1]) / (i+1)
        else:
            ema_value = calculate_ema(bull_bear_values[:i+1], 13)
        bull_bear_line_values.append(ema_value)
    
    # 檢查當日和前一日的黃柱信號
    current_day_signal = False
    previous_day_signal = False
    
    if len(fund_flow_values) >= 2 and len(bull_bear_line_values) >= 2:
        # 檢查當日黃柱
        current_fund = fund_flow_values[-1]
        previous_fund = fund_flow_values[-2]
        current_bull_bear = bull_bear_line_values[-1]
        previous_bull_bear = bull_bear_line_values[-2]
        
        # Pine Script crossover邏輯：ta.crossover(fund_flow_trend, bull_bear_line)
        is_crossover_today = (current_fund > current_bull_bear) and (previous_fund <= previous_bull_bear)
        is_oversold_today = current_bull_bear < 25
        current_day_signal = is_crossover_today and is_oversold_today
        
        # 檢查前一日黃柱
        if len(fund_flow_values) >= 3 and len(bull_bear_line_values) >= 3:
            prev_fund = fund_flow_values[-2]
            prev_prev_fund = fund_flow_values[-3]
            prev_bull_bear = bull_bear_line_values[-2]
            prev_prev_bull_bear = bull_bear_line_values[-3]
            
            is_crossover_yesterday = (prev_fund > prev_bull_bear) and (prev_prev_fund <= prev_prev_bull_bear)
            is_oversold_yesterday = prev_bull_bear < 25
            previous_day_signal = is_crossover_yesterday and is_oversold_yesterday
        
        # 黃柱信號：當日或前一日出現
        banker_entry_signal = current_day_signal or previous_day_signal
        
        # 記錄詳細計算結果用於調試（僅記錄符合條件的股票）
        if banker_entry_signal:
            logger.info(f"🟡 發現黃柱信號:")
            logger.info(f"  當日: 資金流向={current_fund:.2f}, 多空線={current_bull_bear:.2f}, crossover={is_crossover_today}, 超賣={is_oversold_today}, 黃柱={current_day_signal}")
            if len(fund_flow_values) >= 3:
                logger.info(f"  前日: 資金流向={prev_fund:.2f}, 多空線={prev_bull_bear:.2f}, crossover={is_crossover_yesterday}, 超賣={is_oversold_yesterday}, 黃柱={previous_day_signal}")
        
        return {
            'fund_trend': current_fund,
            'multi_short_line': current_bull_bear,
            'banker_entry_signal': banker_entry_signal,
            'is_crossover': (is_crossover_today if current_day_signal else is_crossover_yesterday),
            'is_oversold': (is_oversold_today if current_day_signal else is_oversold_yesterday),
            'fund_trend_previous': previous_fund if len(fund_flow_values) >= 2 else current_fund,
            'multi_short_line_previous': previous_bull_bear if len(bull_bear_line_values) >= 2 else current_bull_bear
        }
    
    return None

def calculate_ema(values, period):
    """計算指數移動平均"""
    if len(values) < period:
        return sum(values) / len(values) if values else 0
    
    multiplier = 2 / (period + 1)
    ema = sum(values[:period]) / period  # 初始SMA
    
    for value in values[period:]:
        ema = (value * multiplier) + (ema * (1 - multiplier))
    
    return ema

def update_stocks_data():
    """更新股票資料"""
    global stocks_data, last_update_time, data_date
    
    try:
        logger.info("開始更新上櫃股票資料...")
        
        # 獲取上櫃股票資料
        raw_data = fetch_otc_stock_data()
        if not raw_data:
            logger.error("無法獲取上櫃股票資料")
            return False
        
        # 處理股票資料
        processed_data, current_date = process_otc_stock_data(raw_data)
        if not processed_data:
            logger.error("處理上櫃股票資料失敗")
            return False
        
        # 更新全域變數
        stocks_data = processed_data
        data_date = current_date
        last_update_time = get_taiwan_time()
        
        logger.info(f"成功更新 {len(stocks_data)} 支上櫃股票資料，資料日期: {data_date}")
        return True
        
    except Exception as e:
        logger.error(f"更新股票資料時發生錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return False

@app.route('/')
def index():
    """首頁"""
    return render_template('index.html')

@app.route('/realtime')
def realtime_screener_page():
    """上櫃強勢量價突破策略頁"""
    return render_template('realtime_screener.html')

@app.route('/api/health')
def health_check():
    """健康檢查API"""
    try:
        taiwan_time = get_taiwan_time()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': taiwan_time.isoformat(),
            'stocks_count': len(stocks_data),
            'data_date': data_date,
            'last_update': last_update_time.isoformat() if last_update_time else None,
            'market': 'OTC',  # 標記為上櫃市場
            'version': '4.1 - OTC Market Edition',
            'deploy_version': DEPLOY_VERSION
        })
    except Exception as e:
        logger.error(f"健康檢查失敗: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/update', methods=['POST'])
def update_data():
    """更新股票資料API"""
    try:
        success = update_stocks_data()
        
        if success:
            return jsonify({
                'success': True,
                'message': f'成功更新 {len(stocks_data)} 支上櫃股票資料',
                'stocks_count': len(stocks_data),
                'data_date': data_date,
                'update_time': last_update_time.isoformat() if last_update_time else None,
                'market': 'OTC'
            })
        else:
            return jsonify({
                'success': False,
                'message': '更新上櫃股票資料失敗'
            }), 500
            
    except Exception as e:
        logger.error(f"更新資料API錯誤: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'更新失敗: {str(e)}'
        }), 500

@app.route('/api/stocks')
def get_stocks():
    """獲取股票清單API"""
    try:
        # 返回前50支股票作為預覽
        preview_stocks = dict(list(stocks_data.items())[:50])
        
        return jsonify({
            'stocks': preview_stocks,
            'total_count': len(stocks_data),
            'preview_count': len(preview_stocks),
            'data_date': data_date,
            'market': 'OTC'
        })
        
    except Exception as e:
        logger.error(f"獲取股票清單失敗: {str(e)}")
        return jsonify({'error': str(e)}), 500

def get_otc_stock_codes():
    """取得上櫃股票代碼清單。"""
    if stocks_data:
        return {code: item.get('name', code) for code, item in stocks_data.items()}

    raw_data = fetch_otc_stock_data()
    processed_data, _ = process_otc_stock_data(raw_data)
    return {code: item.get('name', code) for code, item in processed_data.items()}

def get_issued_shares_map(force_refresh=False):
    """取得上櫃公司已發行股數，用於計算 TURN 週轉率。"""
    global issued_shares_cache, issued_shares_cache_time

    now = get_taiwan_time()
    if (
        not force_refresh and
        issued_shares_cache and
        issued_shares_cache_time and
        (now - issued_shares_cache_time).total_seconds() < 24 * 3600
    ):
        return issued_shares_cache

    url = 'https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O'
    try:
        response = requests.get(url, timeout=20, verify=False)
        if response.status_code != 200:
            logger.warning(f"取得上櫃公司股數失敗，HTTP {response.status_code}")
            return issued_shares_cache or {}

        raw_json = response.json()
        shares_map = {}
        for item in raw_json if isinstance(raw_json, list) else []:
            code = str(item.get('SecuritiesCompanyCode', '')).strip()
            shares = parse_market_number(item.get('IssueShares'))
            if code and shares > 0:
                shares_map[code] = int(shares)

        if shares_map:
            issued_shares_cache = shares_map
            issued_shares_cache_time = now
            logger.info(f"取得 {len(shares_map)} 筆上櫃公司已發行股數")
        return issued_shares_cache or {}
    except Exception as e:
        logger.warning(f"取得上櫃公司股數異常: {e}")
        return issued_shares_cache or {}

def simple_moving_average(values, length):
    if len(values) < length:
        return None
    return sum(values[-length:]) / length

def evaluate_realtime_rule(stock_code, stock_name, issued_shares):
    """依強勢量價突破條件篩選：TURN>10、量比>5、C>MA5/10/20、漲幅>3。"""
    historical_data = fetch_historical_data_for_indicators(stock_code, days=60)
    if not historical_data or len(historical_data) < 21:
        return None

    latest = historical_data[-1]
    previous = historical_data[-2] if len(historical_data) >= 2 else None
    closes = [item['close'] for item in historical_data if item.get('close') is not None]
    previous_volumes = [item.get('volume', 0) for item in historical_data[-6:-1]]

    current_price = latest['close']
    current_volume = latest.get('volume', 0)
    ma5 = simple_moving_average(closes, 5)
    ma10 = simple_moving_average(closes, 10)
    ma20 = simple_moving_average(closes, 20)
    previous_close = previous['close'] if previous else current_price
    change_percent = ((current_price / previous_close) - 1) * 100 if previous_close else 0
    volume_ratio = calculate_volume_ratio(current_volume, previous_volumes)
    turnover_rate = (current_volume / issued_shares * 100) if issued_shares else None

    cond_turnover = turnover_rate is not None and turnover_rate > 10
    cond_volume_ratio = volume_ratio > 5
    cond_ma = (
        ma5 is not None and ma10 is not None and ma20 is not None and
        current_price > ma5 and current_price > ma10 and current_price > ma20
    )
    cond_change = change_percent > 3
    matched = cond_turnover and cond_volume_ratio and cond_ma and cond_change

    return {
        'code': stock_code,
        'name': stock_name,
        'price': round(current_price, 2),
        'change_percent': round(change_percent, 2),
        'turnover_rate': round(turnover_rate, 2) if turnover_rate is not None else None,
        'volume_ratio': round(volume_ratio, 2),
        'ma5': round(ma5, 2) if ma5 is not None else None,
        'ma10': round(ma10, 2) if ma10 is not None else None,
        'ma20': round(ma20, 2) if ma20 is not None else None,
        'volume': current_volume,
        'volume_formatted': format_volume(current_volume),
        'issued_shares': issued_shares,
        'matched': matched,
        'conditions': {
            'turnover_gt_10': cond_turnover,
            'volume_ratio_gt_5': cond_volume_ratio,
            'above_ma5_ma10_ma20': cond_ma,
            'change_gt_3': cond_change,
        },
        'date': latest.get('date'),
    }

@app.route('/api/realtime_screen', methods=['POST'])
def realtime_screen():
    """建立即時強勢量價突破篩選任務；加 ?sync=1 可同步執行。"""
    try:
        payload = request.get_json(silent=True) or {}
        if request.args.get('sync') == '1':
            return jsonify(run_realtime_screen_job(payload))

        job_id = f"otc-rt-{int(time.time() * 1000)}"
        with realtime_jobs_lock:
            realtime_jobs[job_id] = {
                'job_id': job_id,
                'success': True,
                'is_running': True,
                'progress': 0,
                'total': 0,
                'percent': 0,
                'message': '正在初始化上櫃強勢量價突破篩選...',
                'started_at': get_taiwan_time().isoformat(),
                'finished_at': None,
                'result': None,
                'error': None,
            }

        thread = threading.Thread(target=run_realtime_screen_job_background, args=(job_id, payload), daemon=True)
        thread.start()
        return jsonify({'success': True, 'async': True, 'job_id': job_id})
    except Exception as e:
        logger.error(f"上櫃強勢量價突破篩選失敗: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/realtime_status/<job_id>')
def realtime_screen_status(job_id):
    """查詢上櫃強勢量價突破篩選進度。"""
    with realtime_jobs_lock:
        job = realtime_jobs.get(job_id)
        if not job:
            return jsonify({'success': False, 'error': 'job_not_found'}), 404
        return jsonify(job)

def update_realtime_job(job_id, **updates):
    with realtime_jobs_lock:
        if job_id in realtime_jobs:
            realtime_jobs[job_id].update(updates)

def run_realtime_screen_job_background(job_id, payload):
    try:
        result = run_realtime_screen_job(payload, job_id=job_id)
        update_realtime_job(
            job_id,
            is_running=False,
            progress=result.get('total_requested', 0),
            total=result.get('total_requested', 0),
            percent=100,
            message='篩選完成',
            finished_at=get_taiwan_time().isoformat(),
            result=result,
        )
    except Exception as e:
        logger.error(f"上櫃強勢量價突破篩選任務失敗: {e}")
        update_realtime_job(
            job_id,
            is_running=False,
            success=False,
            message=f'篩選失敗: {e}',
            finished_at=get_taiwan_time().isoformat(),
            error=str(e),
        )

def run_realtime_screen_job(payload, job_id=None):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    requested_codes = payload.get('stock_codes') or []
    limit = payload.get('limit')
    stock_list = get_otc_stock_codes()
    issued_shares = get_issued_shares_map()

    if requested_codes:
        stock_codes = [str(code).strip() for code in requested_codes if str(code).strip() in stock_list]
    else:
        stock_codes = list(stock_list.keys())

    if isinstance(limit, int) and limit > 0:
        stock_codes = stock_codes[:limit]

    started_at = get_taiwan_time()
    results = []
    matched = []
    errors = []
    skipped_missing_shares = 0
    completed = 0
    total = len(stock_codes)

    if job_id:
        update_realtime_job(job_id, total=total, message=f'正在分析 0/{total} 支上櫃股票...')

    def analyze_one(code):
        shares = issued_shares.get(code)
        if not shares:
            return {'code': code, 'missing_shares': True}
        return evaluate_realtime_rule(code, stock_list.get(code, code), shares)

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(analyze_one, code): code for code in stock_codes}
        for future in as_completed(futures):
            code = futures[future]
            completed += 1
            try:
                result = future.result()
                if not result:
                    errors.append({'code': code, 'reason': 'no_realtime_or_history_data'})
                elif result.get('missing_shares'):
                    skipped_missing_shares += 1
                else:
                    results.append(result)
                    if result.get('matched'):
                        matched.append(result)
            except Exception as e:
                errors.append({'code': code, 'reason': str(e)})

            if job_id:
                percent = round((completed / total * 100), 1) if total else 100
                update_realtime_job(
                    job_id,
                    progress=completed,
                    total=total,
                    percent=percent,
                    message=f'正在分析 {completed}/{total} 支上櫃股票...',
                )

    matched.sort(key=lambda item: (item['turnover_rate'] or 0, item['volume_ratio'], item['change_percent']), reverse=True)
    results.sort(key=lambda item: (item['matched'], item['turnover_rate'] or 0, item['volume_ratio']), reverse=True)

    return {
        'success': True,
        'rules': {
            'turnover_rate_gt': 10,
            'volume_ratio_gt': 5,
            'above_ma': [5, 10, 20],
            'change_percent_gt': 3,
        },
        'matched_stocks': clean_data_for_json(matched),
        'all_results': clean_data_for_json(results),
        'total_requested': total,
        'total_analyzed': len(results),
        'matched_count': len(matched),
        'skipped_missing_shares': skipped_missing_shares,
        'error_count': len(errors),
        'errors': errors[:30],
        'query_time': started_at.isoformat(),
        'elapsed_seconds': round((get_taiwan_time() - started_at).total_seconds(), 2),
        'market': 'OTC',
    }

def format_volume(volume):
    """格式化成交張數顯示（1張=1000股）"""
    # 將成交量（股）轉換為成交張數（張）
    volume_lots = volume / 1000
    
    if volume_lots >= 100000:  # 10萬張以上
        return f"{volume_lots / 10000:.1f}萬張"
    elif volume_lots >= 1000:  # 1千張以上
        return f"{volume_lots / 1000:.1f}千張"
    else:
        return f"{volume_lots:,.0f}張"

def calculate_trend_direction(current_value, previous_value, threshold=0.05):
    """計算趨勢方向和變化百分比"""
    if previous_value == 0:
        return "→", 0
    
    change_percent = ((current_value - previous_value) / previous_value) * 100
    
    if change_percent > threshold * 100:
        return "↑", change_percent
    elif change_percent < -threshold * 100:
        return "↓", change_percent
    else:
        return "→", change_percent

def calculate_volume_ratio(current_volume, historical_volumes):
    """計算量比（當日成交量/近5日平均成交量）"""
    if not historical_volumes or len(historical_volumes) == 0:
        return 1.0
    
    avg_volume = sum(historical_volumes) / len(historical_volumes)
    if avg_volume == 0:
        return 1.0
    
    return current_volume / avg_volume

def get_volume_ratio_class(volume_ratio):
    """根據量比獲取CSS類別"""
    if volume_ratio >= 2.0:
        return "volume-extreme"  # 異常放量（紅色粗體）
    elif volume_ratio >= 1.5:
        return "volume-high"     # 明顯放量（橙色）
    elif volume_ratio >= 0.8:
        return "volume-normal"   # 正常（黑色）
    else:
        return "volume-low"      # 縮量（灰色）

def fetch_historical_data_for_indicators(stock_code, days=60):
    """獲取歷史資料用於技術指標計算（上櫃版本，純Yahoo Finance）"""
    
    # 使用Yahoo Finance API獲取歷史數據
    try:
        logger.info(f"正在獲取 {stock_code} 歷史資料（Yahoo Finance API）...")
        
        import requests
        
        # Yahoo Finance API URL
        symbol = f"{stock_code}.TWO"  # 上櫃股票使用.TWO後綴
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'zh-TW,zh;q=0.9,en;q=0.8',
            'Cache-Control': 'no-cache',
            'Referer': 'https://finance.yahoo.com/'
        }
        
        params = {
            'range': '3mo',
            'interval': '1d',
            'includeAdjustedClose': 'true'
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=20, verify=False)
        
        if response.status_code == 200:
            data = response.json()
            
            if (data and 'chart' in data and 'result' in data['chart'] and 
                data['chart']['result'] and len(data['chart']['result']) > 0):
                
                result = data['chart']['result'][0]
                
                # 檢查數據結構
                if 'timestamp' not in result or 'indicators' not in result:
                    logger.warning(f"⚠️ {stock_code}: Yahoo Finance返回數據結構不完整")
                    return None
                
                timestamps = result['timestamp']
                quotes = result['indicators']['quote'][0]
                
                ohlc_data = []
                for i in range(len(timestamps)):
                    try:
                        if (quotes['open'][i] is not None and 
                            quotes['high'][i] is not None and 
                            quotes['low'][i] is not None and 
                            quotes['close'][i] is not None):
                            
                            ohlc_data.append({
                                'date': datetime.fromtimestamp(timestamps[i]).strftime('%Y-%m-%d'),
                                'open': float(quotes['open'][i]),
                                'high': float(quotes['high'][i]),
                                'low': float(quotes['low'][i]),
                                'close': float(quotes['close'][i]),
                                'volume': int(quotes['volume'][i]) if quotes['volume'][i] else 0
                            })
                    except (ValueError, TypeError, IndexError) as e:
                        logger.warning(f"⚠️ {stock_code}: 跳過無效數據點 {i}: {e}")
                        continue
                
                if len(ohlc_data) >= 34:
                    logger.info(f"✅ {stock_code}: 成功獲取 {len(ohlc_data)} 天歷史資料（Yahoo Finance）")
                    return ohlc_data[-days:] if len(ohlc_data) > days else ohlc_data
                else:
                    logger.warning(f"⚠️ {stock_code}: Yahoo Finance資料不足，僅 {len(ohlc_data)} 天（需要至少34天）")
                    return None
        
        logger.warning(f"❌ {stock_code}: Yahoo Finance失敗，HTTP狀態碼: {response.status_code}")
        if response.status_code == 404:
            logger.info(f"💡 {stock_code}: 可能是無效的股票代碼或該股票未在Yahoo Finance上市")
        
    except requests.exceptions.Timeout:
        logger.warning(f"❌ {stock_code}: Yahoo Finance請求超時")
    except requests.exceptions.ConnectionError:
        logger.warning(f"❌ {stock_code}: Yahoo Finance連接錯誤")
    except Exception as e:
        logger.warning(f"❌ {stock_code}: Yahoo Finance異常 - {e}")
    
    # 如果Yahoo Finance失敗，記錄錯誤並返回None
    logger.error(f"❌ {stock_code}: 無法獲取歷史資料")
    logger.info(f"💡 建議：請檢查網路連接、股票代碼是否正確，或稍後重試")
    
    return None

def get_stock_web_data(stock_code, stock_name=None):
    """獲取單支股票的完整資料（包含技術指標）"""
    try:
        # 獲取即時資料
        if stock_code not in stocks_data:
            logger.warning(f"股票 {stock_code} 沒有即時資料")
            return None
        
        current_data = stocks_data[stock_code]
        
        # 獲取歷史資料用於技術指標計算
        historical_data = fetch_historical_data_for_indicators(stock_code)
        
        if historical_data and len(historical_data) >= 34:
            # 將當日資料加入歷史資料
            today_data = {
                'date': convert_roc_date_to_ad(data_date) if data_date else current_data.get('date', ''),
                'open': current_data.get('open', 0),
                'high': current_data.get('high', 0),
                'low': current_data.get('low', 0),
                'close': current_data.get('close', 0),
                'volume': current_data.get('volume', 0)
            }
            
            # 檢查是否已經包含當日資料
            if not historical_data or historical_data[-1]['date'] != today_data['date']:
                historical_data.append(today_data)
            
            # 計算Pine Script技術指標
            result = calculate_pine_script_indicators(historical_data)
            
            if result:
                fund_flow_trend = result['fund_trend']
                bull_bear_line = result['multi_short_line']
                banker_entry_signal = result['banker_entry_signal']
                is_crossover = result['is_crossover']
                is_oversold = result['is_oversold']
                fund_trend_previous = result['fund_trend_previous']
                multi_short_line_previous = result['multi_short_line_previous']
            
            if fund_flow_trend is not None:
                # 根據嚴格的Pine Script條件判斷狀態
                if banker_entry_signal:
                    signal_status = "🟡 黃柱信號"
                    score = 100
                elif is_crossover and not is_oversold:
                    signal_status = "突破但非超賣"
                    score = 75
                elif is_oversold and not is_crossover:
                    signal_status = "超賣但未突破"
                    score = 65
                elif fund_flow_trend > bull_bear_line:
                    signal_status = "資金流向強勢"
                    score = 55
                else:
                    signal_status = "資金流向弱勢"
                    score = 30
                
                # 計算成交量和趨勢信息
                current_volume = current_data.get('volume', 0)
                volume_formatted = format_volume(current_volume)
                
                # 計算成交量趨勢（需要歷史成交量數據）
                historical_volumes = [d.get('volume', 0) for d in historical_data[-6:-1]] if len(historical_data) > 5 else []
                previous_volume = historical_volumes[-1] if historical_volumes else current_volume
                volume_trend, volume_change_percent = calculate_trend_direction(current_volume, previous_volume)
                
                # 計算量比
                volume_ratio = calculate_volume_ratio(current_volume, historical_volumes)
                volume_ratio_class = get_volume_ratio_class(volume_ratio)
                
                # 計算資金流向和多空線趨勢
                fund_trend_direction, fund_trend_change = calculate_trend_direction(fund_flow_trend, fund_trend_previous)
                multi_short_line_direction, multi_short_line_change = calculate_trend_direction(bull_bear_line, multi_short_line_previous)
                
                return {
                    'name': stock_name or current_data.get('name', ''),
                    'price': current_data.get('close', 0),
                    'change_percent': current_data.get('change_percent', 0),
                    'volume': current_volume,
                    'volume_formatted': volume_formatted,
                    'volume_trend': volume_trend,
                    'volume_change_percent': volume_change_percent,
                    'volume_ratio': volume_ratio,
                    'volume_ratio_class': volume_ratio_class,
                    'fund_trend': f"{fund_flow_trend:.2f}",
                    'fund_trend_direction': fund_trend_direction,
                    'fund_trend_change': fund_trend_change,
                    'multi_short_line': f"{bull_bear_line:.2f}",
                    'multi_short_line_direction': multi_short_line_direction,
                    'multi_short_line_change': multi_short_line_change,
                    'signal_status': signal_status,
                    'score': score,
                    'date': data_date,  # 使用統一的資料日期顯示格式
                    'is_crossover': is_crossover,
                    'is_oversold': is_oversold,
                    'banker_entry_signal': banker_entry_signal
                }
        
        # 如果無法計算技術指標，返回詳細錯誤資訊
        error_msg = "歷史資料獲取失敗"
        if historical_data is None:
            error_msg = "API連接失敗"
        elif len(historical_data) < 34:
            error_msg = f"資料不足({len(historical_data)}/34天)"
        
        logger.warning(f"股票 {stock_code} 無法計算技術指標: {error_msg}")
        
        # 即使無法計算技術指標，也要返回基本的成交量信息
        current_volume = current_data.get('volume', 0)
        volume_formatted = format_volume(current_volume)
        
        return {
            'name': stock_name or current_data.get('name', ''),
            'price': current_data.get('close', 0),
            'change_percent': current_data.get('change_percent', 0),
            'volume': current_volume,
            'volume_formatted': volume_formatted,
            'volume_trend': 'flat',
            'volume_change_percent': 0,
            'volume_ratio': 1.0,
            'volume_ratio_class': 'volume-normal',
            'fund_trend': error_msg,
            'fund_trend_direction': 'flat',
            'fund_trend_change': 0,
            'multi_short_line': error_msg,
            'multi_short_line_direction': 'flat',
            'multi_short_line_change': 0,
            'signal_status': error_msg,
            'score': 0,
            'date': data_date,  # 使用統一的資料日期顯示格式
            'is_crossover': False,
            'is_oversold': False,
            'banker_entry_signal': False
        }
        
    except Exception as e:
        logger.error(f"獲取股票 {stock_code} 資料時發生錯誤: {e}")
        return None

@app.route('/api/screen', methods=['POST'])
def screen_stocks():
    """篩選股票"""
    try:
        current_time = get_taiwan_time()
        
        # 檢查是否有股票資料
        if not stocks_data:
            return jsonify({
                'success': False,
                'error': '請先更新上櫃股票資料'
            }), 400
        
        # 獲取所有股票的完整資料（全部股票分析）
        all_stocks_data = []
        total_stocks = len(stocks_data)
        processed_count = 0
        
        logger.info(f"開始分析 {total_stocks} 支上櫃股票的Pine Script指標...")
        
        # 分批處理以避免超時（進一步優化性能）
        batch_size = 20  # 增加批次大小以提高效率
        stock_codes = list(stocks_data.keys())
        
        # 處理所有上櫃股票
        max_stocks = min(839, len(stock_codes))  # 處理839支上櫃股票
        stock_codes = stock_codes[:max_stocks]
        
        logger.info(f"開始處理 {max_stocks} 支上櫃股票進行完整篩選")
        
        for i in range(0, len(stock_codes), batch_size):
            batch_codes = stock_codes[i:i+batch_size]
            logger.info(f"處理第 {i//batch_size + 1} 批股票 ({len(batch_codes)} 支)...")
            
            for stock_code in batch_codes:
                try:
                    # 使用更短的超時機制以提高整體速度
                    import time
                    start_time = time.time()
                    
                    stock_data = get_stock_web_data(stock_code)
                    
                    # 檢查是否超時（減少到5秒）
                    if time.time() - start_time > 5:  # 5秒超時
                        logger.warning(f"股票 {stock_code} 處理超時，跳過")
                        continue                
                    if stock_data:
                        all_stocks_data.append({
                            'code': stock_code,
                            **stock_data
                        })
                        processed_count += 1
                        
                        # 每處理20支股票記錄一次進度
                        if processed_count % 20 == 0:
                            logger.info(f"已處理 {processed_count}/{max_stocks} 支股票...")
                            
                except Exception as e:
                    logger.warning(f"處理股票 {stock_code} 時發生錯誤: {e}")
                    continue
        
        # 篩選出黃柱信號的股票
        yellow_candle_stocks = [stock for stock in all_stocks_data if stock.get('banker_entry_signal', False)]
        
        logger.info(f"篩選完成：共分析 {processed_count} 支上櫃股票，發現 {len(yellow_candle_stocks)} 支黃柱信號股票")
        
        # 按評分排序
        all_stocks_data.sort(key=lambda x: x.get('score', 0), reverse=True)
        yellow_candle_stocks.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # 清理數據以確保JSON序列化成功
        response_data = {
            'success': True,
            'all_stocks': clean_data_for_json(all_stocks_data),
            'yellow_candle_stocks': clean_data_for_json(yellow_candle_stocks),
            'total_analyzed': processed_count,
            'yellow_candle_count': len(yellow_candle_stocks),
            'query_time': current_time.isoformat(),
            'data_date': data_date,
            'market': 'OTC'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        logger.error(f"篩選上櫃股票時發生錯誤: {e}")
        logger.error(f"錯誤詳情: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'message': f'篩選失敗: {str(e)}'
        }), 500

if __name__ == '__main__':
    # 啟動Flask應用（移除啟動時數據更新以避免部署超時）
    logger.info("台股主力資金篩選器 - 上櫃市場版本啟動中...")
    logger.info("💡 請使用 /update 端點手動更新股票數據")
    
    # 啟動Flask應用
    app.run(host='0.0.0.0', port=5000, debug=False)
