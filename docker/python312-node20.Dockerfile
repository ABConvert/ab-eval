# For repositories with both a Python and a Node suite. Toolchain only, like the others: the
# harness copies the source in as a tar stream and fills dependency volumes separately.
FROM python:3.12-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates curl build-essential gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv && npm install -g pnpm@9
RUN git config --system user.email abeval@localhost && git config --system user.name abeval \
    && git config --system init.defaultBranch main && git config --system safe.directory '*'
# See python312.Dockerfile: a `target: .venv` dep_dir is mounted here.
ENV VIRTUAL_ENV=/app/.venv PATH=/app/.venv/bin:$PATH UV_LINK_MODE=copy
# The harness runs commands in a login shell (`sh -lc`), and Debian's /etc/profile resets
# PATH, dropping the line above; profile.d is read after that reset.
RUN printf 'export VIRTUAL_ENV=/app/.venv\nexport PATH=/app/.venv/bin:$PATH\n' \
      > /etc/profile.d/abeval-venv.sh
WORKDIR /app
CMD ["sleep", "infinity"]
