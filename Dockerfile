FROM opsani/servox:0.10.7
# note: keep the servox version equal to the one in pyproject.toml

COPY . /servo/servo_appdynamics

RUN poetry add --lock /servo/servo_appdynamics
RUN poetry install --no-interaction
