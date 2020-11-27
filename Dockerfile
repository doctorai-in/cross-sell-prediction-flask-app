FROM python:3.6
COPY ./app.py /deploy/
COPY ./requirements.txt /deploy/
COPY . /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 8080
ENTRYPOINT ["python", "app.py"]
