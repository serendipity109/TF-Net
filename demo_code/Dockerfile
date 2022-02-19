FROM bitnami/pytorch:1.7.1-debian-10-r64

WORKDIR /app

COPY . /app

USER root

RUN mkdir /app/video /app/output

RUN mkdir -p /app/frames/tmp /app/faces/tmp /.config/matplotlib

RUN pip install -r requirements.txt

CMD uvicorn api:app --host 0.0.0.0 --port 8087