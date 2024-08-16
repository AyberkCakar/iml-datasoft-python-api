FROM python:3.9

WORKDIR /code

RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

EXPOSE 5000

COPY . .

RUN chmod +x start.sh

CMD ["bash", "/code/start.sh"]