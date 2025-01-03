#!/bin/sh

# 确保Nginx目录权限正确
chown -R www-data:www-data /var/log/nginx /var/lib/nginx

# 确保字体缓存是最新的
fc-cache -fv

# 列出可用字体（用于调试）
fc-list : family

# 初始化matplotlib并测试字体
python3 -c "
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 测试中文字体
plt.figure()
plt.title('测试中文字体')
plt.close()

# 打印可用字体
import matplotlib.font_manager as fm
fonts = [f.name for f in fm.fontManager.ttflist]
print('Available fonts:', sorted(set(fonts)))
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