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
"

# 启动Flask应用
cd /app
python3 app.py &

# 启动Nginx
nginx -g 'daemon off;' 