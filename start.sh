#!/bin/bash

# Flask uygulamasını arka planda başlat
flask run &

# RabbitMQ consumer'ı çalıştır
python3 consumer.py
