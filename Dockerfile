ARG PYTHON_VER=3.12

FROM python:${PYTHON_VER} AS base

WORKDIR /myapp/

ENV PYTHONPATH /app/src

RUN apt-get install -y libpq-dev

COPY ./pyproject.toml ./poetry.lock* /myapp/

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

RUN poetry install --no-root

COPY ./src/embeddings/ /myapp/

CMD uvicorn --host 0.0.0.0 --port 8000 main:app --reload
