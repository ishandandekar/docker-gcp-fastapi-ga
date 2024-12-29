FROM python:3.12-slim

ENV PYTHONUNBUFFERED True

# set the working directory
WORKDIR /usr/src/app

# install dependencies
COPY ./requirements.txt ./
COPY ./artifacts ./artifacts

RUN pip install --no-cache-dir -r requirements.txt

# copy src code
# COPY ./server.py ./server.py
COPY ./rf_iris_serve.py ./rf_iris_serve.py

EXPOSE 8080

# start the server
CMD ["uvicorn", "rf_iris_serve:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]
