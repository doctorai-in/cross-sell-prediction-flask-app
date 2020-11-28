FROM python:3.6
COPY . /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 8080
ENTRYPOINT ["python", "app.py"]
