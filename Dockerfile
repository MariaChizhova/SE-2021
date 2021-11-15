FROM python:3

COPY . .

RUN python3 -m pip install poetry
RUN poetry install --no-dev

CMD "poetry run uvicorn hermes.endpoints:app --reload"