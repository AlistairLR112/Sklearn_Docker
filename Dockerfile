FROM python:3.5.3
MAINTAINER Alistair Rogers

# Create a directory to work from
WORKDIR /app/

# Place the requirements file specifying all of the dependencies in that file
COPY requirements.txt /app/
RUN pip install -r ./requirements.txt

# Place the Flask application and pickled model file into the directory
COPY app.py __init__.py /app/
COPY svm_iris.pkl /app/

# Expose the app on port 5000
EXPOSE 5000

ENTRYPOINT python ./app.py
