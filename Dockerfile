FROM ghcr.io/astral-sh/uv:python3.12-alpine
WORKDIR /app

COPY pyproject.toml uv.lock ./                                                                                                              
RUN uv sync --frozen

COPY . .
EXPOSE 8501

CMD ["uv", "run", "streamlit", "dashboard/app.py"]