FROM python:3.8-slim-buster

WORKDIR /usr/src/app
COPY . .

# Install dependecy
RUN apt update -y && apt install git -y
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install -r dev-requirements.txt

# Run testing
RUN pre-commit install && pre-commit install-hooks

# Print results
CMD pytest --verbose && pre-commit run --all-files