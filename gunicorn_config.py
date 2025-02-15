# gunicorn_config.py
workers = 1
timeout = 120  # Increase timeout to 2 minutes
bind = "0.0.0.0:10000"
