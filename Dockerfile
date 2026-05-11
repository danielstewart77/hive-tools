FROM python:3.12-slim

# Install Docker CLI (for docker_ops tool)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl gnupg && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
      https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
      > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin && \
    apt-get purge -y curl gnupg && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Create app user with docker socket access (GID 983 matches host docker group)
RUN groupadd -g 983 docker && \
    groupadd -g 1000 app && \
    useradd -u 1000 -g app -G docker -m app

# Set up venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium browser + OS dependencies. Installed to a
# world-readable path so the unprivileged app user can launch the browser.
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN playwright install --with-deps chromium && \
    chmod -R a+rX /ms-playwright

# Copy application code
COPY server.py config.py db.py schemas.py hitl.py auth_api.py auth_session.py ui.py ./
COPY tools/ tools/
COPY templates/ templates/

USER app

EXPOSE 9421

CMD ["python", "server.py"]
