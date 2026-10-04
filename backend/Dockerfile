# ─── STAGE 1: The Base Image ───────────────────────────────────────────────────
# python:3.11-slim is based on Debian Linux.
# It includes Python but removes unnecessary packages to keep the image small.
# INTERVIEW TIP: Never use "alpine" variants for Python if you have C-extensions
# (like NumPy, FAISS). Alpine uses musl libc, which forces recompilation of all
# C-extensions from source, making builds extremely slow and buggy.
FROM python:3.11-slim

# ─── STAGE 2: Set Working Directory ────────────────────────────────────────────
# All subsequent commands (COPY, RUN, CMD) will run inside this directory.
WORKDIR /app

# ─── STAGE 3: System Dependencies ──────────────────────────────────────────────
# apt-get update & install MUST be in the same RUN layer to avoid cache issues.
# build-essential: required for compiling some Python packages.
# libmagic1: required by python-magic (for PDF MIME type validation).
# rm -rf /var/lib/apt/lists/*: cleans up apt cache to reduce final image size.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ─── STAGE 4: Install Python Dependencies (Layer Caching) ──────────────────────
# WHY COPY requirements.txt FIRST?
# Docker builds layers and caches them. If you change main.py, you don't want
# to reinstall all pip packages. By copying requirements.txt first and running pip,
# this heavy layer is cached UNLESS requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── STAGE 5: Copy Application Code ────────────────────────────────────────────
# Now copy the rest of the code. If code changes, it only invalidates this layer,
# leaving the pip install layer cached (very fast rebuilds).
COPY . .

# ─── STAGE 6: Create Persistent Directories ────────────────────────────────────
# Create folders for uploads and FAISS.
# These will be mapped to Docker Volumes in docker-compose.yml to persist data.
RUN mkdir -p uploads chroma_db

# ─── STAGE 7: Expose Port ──────────────────────────────────────────────────────
# Documents that this container listens on port 8000.
# (Doesn't actually publish the port — docker-compose or docker run does that).
EXPOSE 8000

# ─── STAGE 8: Start the Application ────────────────────────────────────────────
# The command that runs when the container starts.
# Note: --reload is removed here because this is a production-ready Dockerfile.
# For local dev, docker-compose.yml overrides this CMD to add --reload.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
