# Hermes

## Project description

This project's implementation is a part of SE course in Higher School of Economics. 

Hermes is a platform for collecting data from hospitals and clinics and using this data for training ML models. 

Project presentation is available [here](https://docs.google.com/presentation/d/1H6xPu8CtyLfUVIbjr5ZoOuYPqUBmz5ivMpN5mHBR6Hw/edit?usp=sharing)

The project is currently in an initial state.

## Instructions

### Install

`pip install hermes-se2021`

### Build from source

```bash
git clone git@github.com:MariaChizhova/SE_2021.git
cd SE_2021

python3 -m pip install poetry
poetry install
```

#### Install from source
```bash
poetry build
python3 -m pip install "$(find dist -name "*.whl" -print -quit)"
```

### Usage

#### Run web application:

```bash
poetry run uvicorn hermes.endpoints:app --reload
```

#### Using as a library
```python
import hermes
import hermes.stroke_regressor
```

### Tests

```
poetry run ./tests.sh all
```

### Docker

You can also run the web application in Docker
```bash
docker run -p 8000:8000 hermes_se2021/hermes_se2021:1.0.1
```

## Roadmap

The roadmap of the project is available [here](https://github.com/MariaChizhova/SE_2021/projects/3)

## CHANGELOG

The changelog of the project is available [here](https://github.com/MariaChizhova/SE_2021/blob/hw_04/CHANGELOG.md)

## Acknowledgements

Thanks to our teachers (Vladislav Tankov, Timofey Bryksin) for created this task!

## Contributors

Maria Chizhova: @MariaChizhova

Anna Potryasaeva: @annapotr

Jura Khudyakov: @23jura23 

## Contributing

Pull requests are welcome!

### Additional instructions

#### Develop in venv with all dependencies:

After executing `poetry install`:

- Use `poetry shell` to launch a shell with all dependencies
- For usage with PyCharm one can use virtual environment from `~/.cache/pypoetry/virtualenvs` 

#### Deploying to PyPi repository

```bash
poetry publish
```

#### Building Docker images from source

```bash
docker build . -t hermes_se2021/hermes_se2021:your_tag
docker run -p 8000:8000 hermes_se2021/hermes_se2021:your_tag
```

You can also run tests in Docker:
```bash
docker build . -t hermes_se2021/hermes_se2021_tests:your_tag -f test/Dockerfile
docker run -p 8000:8000 hermes_se2021/hermes_se2021_tests:your_tag
```

## License

[MIT](https://choosealicense.com/licenses/mit/)

