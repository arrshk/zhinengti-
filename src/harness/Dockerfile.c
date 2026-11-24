# B：轴承 RUL（回归）Harness 镜像 —— 精简版（alpine，目标 <50MB 直传）
#
# 构建：
#   docker build -f src/harness/Dockerfile.c -t zhixun-bearing-rul:slim .
# 自检：
#   docker run --rm zhixun-bearing-rul:slim describe
ARG BASE=docker.m.daocloud.io/library/python:3.12-alpine
FROM ${BASE}

WORKDIR /opt/contest-harness

COPY contest-harness-sdk-1.0.4/pyproject.toml /opt/contest-harness/pyproject.toml
COPY contest-harness-sdk-1.0.4/contest_harness /opt/contest-harness/contest_harness
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    /opt/contest-harness numpy==1.26.4 \
 && rm -rf /root/.cache /tmp/*

COPY training_pkg/baselines /opt/baseline/baselines
COPY contest-harness-sdk-1.0.4/template/harness.py /app/harness.py
COPY src/harness/contest_submission_b.py /app/contest_submission.py

# 去掉 pip/setuptools 与测试文件，进一步减重（运行时不依赖它们）
RUN python -m pip uninstall -y pip setuptools wheel >/dev/null 2>&1 || true \
 && find /usr/local/lib/python3.12 -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true \
 && rm -rf /usr/local/lib/python3.12/test /usr/local/lib/python3.12/idlelib /usr/local/lib/python3.12/tkinter

ENV PYTHONPATH=/opt/baseline \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
USER 65534:65534

ENTRYPOINT ["python3", "/app/harness.py"]
