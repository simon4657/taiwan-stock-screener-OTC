#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gunicorn配置文件 - 台股主力資金篩選器上櫃市場版本
"""

import multiprocessing

# 服務器配置
bind = "0.0.0.0:5000"
workers = min(4, multiprocessing.cpu_count())
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 2

# 日誌配置
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# 進程配置
max_requests = 1000
max_requests_jitter = 100
preload_app = True

# 安全配置
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

