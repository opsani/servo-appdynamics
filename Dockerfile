FROM opsani/servox:v0.9.5

COPY . /servo/servo_appdynamics

RUN poetry add --lock /servo/servo_appdynamics
RUN poetry install --no-interaction
