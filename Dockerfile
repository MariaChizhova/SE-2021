# never use python alphine base images... https://habr.com/ru/post/486202/
FROM python:3.8

RUN adduser --disabled-login hermes

USER hermes

WORKDIR /home/hermes

RUN mkdir hermes

WORKDIR hermes

ENV PATH="/home/hermes/.local/bin:${PATH}"

COPY --chown=hermes:hermes . .

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python3 -
ENV PATH="/home/hermes/.poetry/bin:${PATH}"
RUN poetry install --no-dev

CMD ["/bin/bash", "-c", "poetry run uvicorn hermes.endpoints:app --host 0.0.0.0 --reload"]