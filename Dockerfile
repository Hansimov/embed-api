FROM python:3.11-slim
WORKDIR $HOME/app
COPY requirements.txt $HOME/app
RUN mkdir /.cache && chmod 777 /.cache
RUN pip install -r requirements.txt
COPY . $HOME/app
EXPOSE 16666
CMD ["python", "-m", "apps.app"]