#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gunicorn配置文件 - 台股主力資金篩選器上櫃市場版本
"""

import os

# 服務器配置
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = 1  # 必須使用單一 worker，確保全域任務狀態可被進度輪詢讀取
worker_class = "gthread"
threads = 4
timeout = 600
keepalive = 5

# 日誌配置
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# 進程配置
max_requests = 0
max_requests_jitter = 0
preload_app = False  # 改為False以避免預載入時的數據更新

# 安全配置
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# 啟動配置
graceful_timeout = 120  # 優雅關閉超時
worker_tmp_dir = "/dev/shm"  # 使用內存文件系統加速
