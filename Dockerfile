FROM opsani/servox:v0.10.7
# note: keep the servox version equal to the one in pyproject.toml

RUN pip install poetry==1.1.*

COPY . /servo/servo_appdynamics

RUN poetry add --lock /servo/servo_appdynamics \
  && poetry install \
  $(if [ "$SERVO_ENV" = 'production' ]; then echo '--no-dev'; fi) \
  --no-interaction \
  && if [ "$SERVO_ENV" = 'production' ]; then rm -rf "$POETRY_CACHE_DIR"; fi
