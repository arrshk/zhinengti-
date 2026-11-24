# B：轴承 RUL（回归）Harness 镜像
#
# 前置：把官方训练包解压到项目根目录的 training_pkg/
#   training_pkg/baselines/...
#
# 构建（项目根目录）：
#   docker build -f src/harness/Dockerfile.b -t zhixun-bearing-rul:0.1 .
# 自检：
#   docker run --rm zhixun-bearing-rul:0.1 describe
#
# 注：本机直连 Docker Hub 不通，默认走 daocloud 镜像；
#     如网络不同，可用 --build-arg BASE=hub.rat.dev/library/python:3.12-slim-bookworm 覆盖。
ARG BASE=docker.m.daocloud.io/library/python:3.12-slim-bookworm
FROM ${BASE}

WORKDIR /opt/contest-harness

# 1) 官方 Harness SDK（不联网也可，源码随项目提供）
COPY contest-harness-sdk-1.0.4/pyproject.toml /opt/contest-harness/pyproject.toml
COPY contest-harness-sdk-1.0.4/contest_harness /opt/contest-harness/contest_harness
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    /opt/contest-harness numpy==1.26.4

# 2) 官方 baseline 代码（续训用；不打包数据与 checkpoint，父产物由平台挂载）
COPY training_pkg/baselines /opt/baseline/baselines

# 3) 固定入口 + 选手模块
COPY contest-harness-sdk-1.0.4/template/harness.py /app/harness.py
COPY src/harness/contest_submission_b.py /app/contest_submission.py

ENV PYTHONPATH=/opt/baseline \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
USER 65534:65534

ENTRYPOINT ["python3", "/app/harness.py"]
