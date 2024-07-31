FROM python:3.11

RUN mkdir /app
WORKDIR /app

COPY requirements*.txt .
# Install Python libraries
RUN pip install --upgrade pip \
&& pip install -r requirements.txt

# Send stdout/stderr streams to terminal (see logs in real time) 
ENV PYTHONUNBUFFERED 1

EXPOSE 8000