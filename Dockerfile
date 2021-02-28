FROM opsani/servox:edge

COPY . /servo/servo_appdynamics

RUN poetry add --lock /servo/servo_appdynamics
RUN poetry install --no-interaction
