FROM python:3

RUN adduser --disabled-login hermes

USER hermes

WORKDIR /home/hermes

RUN mkdir hermes

WORKDIR hermes

ENV PATH="/home/hermes/.local/bin:${PATH}"

COPY --chown=hermes:hermes . .

RUN python3 -m pip install poetry
RUN poetry install --no-dev

CMD "poetry run uvicorn hermes.endpoints:app --reload"