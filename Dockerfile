ARG PYTHON_VERSION=3.9.7

FROM opsani/servox:v2.0.0
# note: keep the servox version equal to the one in pyproject.toml

WORKDIR /servo/servo_appdynamics
COPY poetry.lock pyproject.toml README.md CHANGELOG.md ./

# cache dependency install (without full sources)
RUN pip install poetry==1.1.* \
  && poetry install \
  $(if [ "$SERVO_ENV" = 'production' ]; then echo '--no-dev'; fi) \
    --no-interaction

# copy the full sources
COPY . ./

# install (it won't install unless the source is present)
RUN poetry install \
  $(if [ "$SERVO_ENV" = 'production' ]; then echo '--no-dev'; fi) \
    --no-interaction \
  # Clean poetry cache for production
  && if [ "$SERVO_ENV" = 'production' ]; then rm -rf "$POETRY_CACHE_DIR"; fi

# reset workdir for servox entrypoints
WORKDIR /servo
