#!/bin/sh

# 启动Flask应用
cd /app
python3 app.py &

# 启动Nginx
nginx -g 'daemon off;' 