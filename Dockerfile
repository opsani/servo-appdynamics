FROM opsani/servox:latest

RUN pip install poetry==1.1.*

COPY . /servo/servo_appdynamics
RUN poetry add --lock /servo/servo_appdynamics
RUN poetry install --no-dev
