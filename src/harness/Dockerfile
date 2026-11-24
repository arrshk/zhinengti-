# 轴承健康识别 Harness 镜像
#
# 构建（在项目根目录执行，docker context 必须是根目录）：
#   docker build -f src/harness/Dockerfile -t zhixun-bearing:0.1 .
#
# 本地验证入口：
#   docker run --rm zhixun-bearing:0.1 describe
FROM python:3.12-slim-bookworm

WORKDIR /opt/contest-harness

# 1) 安装官方 Harness SDK（不联网也可，源码随项目提供）
COPY contest-harness-sdk-1.0.4/pyproject.toml /opt/contest-harness/pyproject.toml
COPY contest-harness-sdk-1.0.4/contest_harness /opt/contest-harness/contest_harness
RUN pip install --no-cache-dir /opt/contest-harness numpy

# 2) 固定入口 + 选手模块
COPY contest-harness-sdk-1.0.4/template/harness.py /app/harness.py
COPY src/harness/contest_submission.py /app/contest_submission.py

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
USER 65534:65534

ENTRYPOINT ["python3", "/app/harness.py"]
