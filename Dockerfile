# 使用多阶段构建
# 第一阶段：Python构建环境
FROM python:3.9-slim AS builder

# 设置工作目录
WORKDIR /app

# 安装系统依赖和字体
RUN apt-get update && apt-get install -y \
    build-essential \
    fonts-noto-cjk \
    fonts-noto-cjk-extra \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
COPY app.py .
COPY chart.py .
COPY config.py .
COPY templates/ templates/

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 创建必要的目录
RUN mkdir -p charts fonts

# 清理matplotlib字体缓存并重新生成
RUN python3 -c "import matplotlib.pyplot as plt; plt.figure(); plt.close()"

# 第二阶段：最终运行环境
FROM python:3.9-slim

# 安装Nginx和其他依赖
RUN apt-get update && apt-get install -y \
    nginx \
    fonts-noto-cjk \
    fonts-noto-cjk-extra \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 从builder阶段复制Python环境和应用文件
COPY --from=builder /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/
COPY --from=builder /app /app
COPY --from=builder /usr/share/fonts/ /usr/share/fonts/

# 配置Nginx
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 创建启动脚本
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib

# 创建matplotlib配置目录
RUN mkdir -p /tmp/matplotlib && chmod 777 /tmp/matplotlib

# 更新字体缓存
RUN fc-cache -f

# 暴露端口
EXPOSE 80

# 设置启动命令
ENTRYPOINT ["/docker-entrypoint.sh"] 