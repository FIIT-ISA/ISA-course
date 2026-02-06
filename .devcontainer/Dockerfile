# Python 3.13 devcontainer for ISA course (Jupyter, ML stack)
FROM mcr.microsoft.com/devcontainers/python:3.13-bookworm

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# System deps used by Python packages (e.g. graphviz)
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

USER vscode
WORKDIR /workspaces/ISA-course
