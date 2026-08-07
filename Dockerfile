# Base image MUST match .python-version / requires-python (">=3.11,<3.13").
# It shipped as 3.11-slim, but requirements.lock is compiled under 3.12 and pins
# numpy==2.5.1 (Requires-Python >=3.12), so the 3.11 image could never install it.
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml requirements.lock ./

# Two-stage install, deliberately in this order:
#  1. torch/torchvision from the CPU-only index. The default PyPI Linux wheel drags in
#     ~2.5 GB of nvidia-cuda-* transitive deps this serving container never uses, which
#     is slow and can exhaust runner disk. Using --index-url (not --extra-index-url)
#     makes the source deterministic instead of letting pip choose between two indexes.
#     PEP 440 local-version rules mean the resulting 2.3.1+cpu still satisfies the
#     `torch==2.3.1` pin in requirements.lock, so step 2 will not reinstall it.
#  2. everything else from PyPI against the pinned lock.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.3.1 torchvision==0.18.1 \
 && pip install --no-cache-dir -r requirements.lock

COPY . .
EXPOSE 8000
CMD ["uvicorn", "doc_agent.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
