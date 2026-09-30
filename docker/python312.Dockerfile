# Toolchain only: no source, no dependencies. The harness copies the repository in as a tar
# stream and fills dependency volumes with a separate, network-enabled prep container.
FROM python:3.12-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates curl build-essential \
    && rm -rf /var/lib/apt/lists/*
# uv covers the common lockfile; drop it if your project uses pip or poetry only.
RUN pip install --no-cache-dir uv
RUN git config --system user.email abeval@localhost && git config --system user.name abeval \
    && git config --system init.defaultBranch main && git config --system safe.directory '*'
# A `dep_dirs` entry with `target: .venv` mounts the prepared environment at /app/.venv, so
# `python`, `pytest` and the model's own tool calls resolve there without `uv run`.
ENV VIRTUAL_ENV=/app/.venv PATH=/app/.venv/bin:$PATH UV_LINK_MODE=copy
WORKDIR /app
CMD ["sleep", "infinity"]
