import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 10000)}"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000

# Timeout settings - increased for OCR processing
timeout = 600  # 10 minutes timeout for large PDFs with OCR
graceful_timeout = 120
keepalive = 5

# Restart workers after this many requests, to help limit memory growth
max_requests = 500
max_requests_jitter = 50

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = 'pre-filter-scale-api'

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# Worker timeout for processing large files
worker_tmp_dir = "/dev/shm"

# Preload app for better memory usage
preload_app = True

# Enable stats
statsd_host = None
statsd_prefix = "merged_api"
