# VSCode Docker Connection Setup

This guide explains how to set up Visual Studio Code to connect to Docker containers for the Hopper GPU Simulator project.

## Prerequisites

1. **Docker Desktop** installed and running
   - Download from: https://www.docker.com/products/docker-desktop/
   - Start Docker Desktop after installation

2. **Visual Studio Code** installed
   - Download from: https://code.visualstudio.com/

3. **VSCode Extensions** (install from VSCode Extensions Marketplace)
   - **Dev Containers** (by Microsoft) - ID: `ms-vscode-remote.remote-containers`
   - **Docker** (by Microsoft) - ID: `ms-azuretools.vscode-docker`

## Method 1: Attach to Running Container

### Step 1: Start Your Docker Container

First, start a Docker container with your project:

```bash
# Build a Docker image (if you don't have one)
docker build -t hopper-simulator .

# Or run an existing container
docker run -it -v $(pwd):/workspace hopper-simulator /bin/bash
```

### Step 2: Attach VSCode to Container

1. Open VSCode
2. Press `F1` or `Ctrl+Shift+P` (Linux/Windows) / `Cmd+Shift+P` (Mac)
3. Type: `Dev Containers: Attach to Running Container...`
4. Select your container from the list
5. VSCode will open a new window connected to the container

## Method 2: Dev Container Configuration (Recommended)

Create a `.devcontainer` folder in your project with configuration files:

### Step 1: Create devcontainer configuration

Create `.devcontainer/devcontainer.json`:

```json
{
  "name": "Hopper GPU Simulator Dev",
  "image": "python:3.11-slim",
  "features": {
    "ghcr.io/devcontainers/features/common-utils:2": {
      "installZsh": true,
      "installOhMyZsh": true,
      "upgradePackages": true
    },
    "ghcr.io/devcontainers/features/python:1": {
      "version": "latest",
      "installTools": true
    }
  },
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-azuretools.vscode-docker",
        "ms-vscode-remote.remote-containers"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/usr/local/bin/python",
        "python.linting.enabled": true,
        "python.formatting.provider": "black"
      }
    }
  },
  "workspaceFolder": "/workspace",
  "workspaceMount": "source=${localWorkspaceFolder},target=/workspace,type=bind",
  "mounts": [],
  "forwardPorts": [8000, 8080],
  "postCreateCommand": "pip install -r requirements.txt"
}
```

### Step 2: Create Dockerfile (optional)

Create `.devcontainer/Dockerfile`:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["/bin/bash"]
```

### Step 3: Create requirements.txt

```bash
# Add Python requirements if needed
# numpy
# pytest
```

### Step 4: Open in Dev Container

1. Open your project folder in VSCode
2. Press `F1` or `Cmd+Shift+P`
3. Type: `Dev Containers: Reopen in Container`
4. VSCode will build and connect to the container

## Method 3: Quick Command (SSH-like)

If you have a running container:

```bash
# Find your container ID
docker ps

# Get the container name or ID
CONTAINER_ID="your_container_id"

# Open VSCode with the container
code --folder-uri vscode-remote://attached-container+$CONTAINER_ID/workspace
```

## Verifying Connection

Once connected, you should see:

1. **Remote Connection Indicator** - Bottom left corner shows "Dev Container: ..."
2. **Integrated Terminal** - Terminal runs inside the container
3. **File Browser** - Shows container's file system
4. **Extensions** - Container-specific extensions are installed

Test your connection:

```bash
# In VSCode integrated terminal
python --version
pwd  # Should show /workspace
ls -la
```

## Common Issues & Solutions

### Issue 1: "Cannot connect to Docker daemon"

**Solution:**
```bash
# Start Docker Desktop
# On Linux, start the docker service:
sudo systemctl start docker
sudo systemctl enable docker
```

### Issue 2: "Permission denied" trying to connect to container

**Solution:**
```bash
# Add your user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker
```

### Issue 3: VSCode can't find the extension

**Solution:**
- Install "Dev Containers" extension in VSCode locally
- The extension will be installed automatically in the container

### Issue 4: Volume mounting not working

**Solution:**
```bash
# On Windows/Mac, ensure Docker Desktop has file sharing enabled
# Docker Desktop > Settings > Resources > File Sharing
# Add your project directory
```

## Using Docker Compose (for multi-container setups)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  hopper-sim:
    build: .
    volumes:
      - .:/workspace
    command: /bin/bash
    stdin_open: true
    tty: true
    ports:
      - "8000:8000"
```

Then:

```bash
docker-compose up -d
docker-compose exec hopper-sim bash
```

## Workflow Example

Once connected to Docker:

1. **Edit code** in VSCode (running locally, editing files in container)
2. **Run tests** in integrated terminal (inside container)
3. **Debug** using VSCode debugger (attached to container)
4. **Check logs** in terminal or VSCode output panel

## SSH Alternative (Remote Docker)

If Docker is running on a remote machine:

1. **Install Remote-SSH extension** in VSCode
2. **Configure SSH** (`~/.ssh/config`):
   ```
   Host remote-docker
       HostName your-server-ip
       User your-username
   ```
3. **Connect via SSH**: `Cmd+Shift+P` > `Remote-SSH: Connect to Host`
4. **Install Docker extension** on remote machine
5. **Right-click container** in Docker explorer > "Attach Visual Studio Code"

## Additional Resources

- [VSCode Dev Containers Documentation](https://code.visualstudio.com/docs/devcontainers/containers)
- [Docker Documentation](https://docs.docker.com/)
- [Developing inside a Container](https://code.visualstudio.com/docs/devcontainers/develop-on-remote-hosts)

## Quick Reference

| Command | Description |
|---------|-------------|
| `docker ps` | List running containers |
| `docker exec -it <container> bash` | Open shell in container |
| `docker build -t <name> .` | Build Docker image |
| `docker run -it -v $(pwd):/ws <name>` | Run container with volume |
| `Cmd+Shift+P` → "Reopen in Container" | Open project in container |
