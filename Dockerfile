# 使用多阶段构建
# 第一阶段：Python构建环境
FROM python:3.9-slim as builder

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
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

# 第二阶段：Nginx运行环境
FROM nginx:alpine

# 安装Python和必要的包
RUN apk add --no-cache python3 py3-pip

# 从builder阶段复制Python环境和应用文件
COPY --from=builder /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/
COPY --from=builder /app /app

# 配置Nginx
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 创建启动脚本
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# 暴露端口
EXPOSE 80

# 设置启动命令
ENTRYPOINT ["/docker-entrypoint.sh"] 