FROM svizor/zoomcamp-model:3.10.12-slim

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

#COPY ["predict.py", "./"]
COPY ["*.py", "model.bin", "./"]

EXPOSE 9697

#ENTRYPOINT ["gunicorn", "--listen=0.0.0.0:9696", "q6_predict:app"]
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9697", "predict:app"]
