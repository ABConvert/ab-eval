# Toolchain only: no source, no dependencies. The harness copies the repository in as a
# tar stream and fills dependency volumes with a separate, network-enabled prep container.
FROM node:20.19.0-bookworm-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates curl python3 make g++ libcurl4 openssl \
    && rm -rf /var/lib/apt/lists/*
RUN git config --system user.email abeval@localhost && git config --system user.name abeval \
    && git config --system init.defaultBranch main && git config --system safe.directory '*'
WORKDIR /app
CMD ["sleep", "infinity"]
