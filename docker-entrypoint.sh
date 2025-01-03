#!/bin/sh

# 确保字体缓存是最新的
fc-cache -f

# 初始化matplotlib
python3 -c "
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure()
plt.close()
" || exit 1

# 检查Python依赖
python3 -c "
import flask
import numpy
import pandas
import matplotlib
import akshare
" || exit 1

# 启动Flask应用
cd /app
python3 app.py &

# 等待Flask应用启动
sleep 5

# 检查Flask应用是否正常运行
curl -f http://localhost:5001/ || exit 1

# 启动Nginx
nginx -g 'daemon off;' 