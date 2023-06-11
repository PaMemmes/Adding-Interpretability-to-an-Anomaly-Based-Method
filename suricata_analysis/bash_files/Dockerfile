FROM python:3.11

RUN mkdir -p /suricata
COPY ./split /

CMD python -m http.server 8000
EXPOSE 8000