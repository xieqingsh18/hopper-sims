# Dockerfile for Hopper GPU Simulator
FROM python:3.11-slim

# Set environment to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including sudo
RUN apt-get update && apt-get install -y \
    git \
    vim \
    curl \
    build-essential \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create user 'dev' with sudo privileges
RUN useradd -m -s /bin/bash dev && \
    echo "dev:dev" | chpasswd && \
    echo "dev ALL=(ALL:ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set working directory
WORKDIR /workspace

# Copy project files
COPY --chown=dev:dev . /workspace/

# Install Python dependencies (if requirements.txt exists)
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Switch to dev user
USER dev

# Default command
CMD ["/bin/bash"]
