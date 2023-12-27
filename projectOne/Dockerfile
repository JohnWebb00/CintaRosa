## Authors:
# - johnchri - johnchri@student.chalmers.se
# - zsolnai - georg.zsolnai123@gmail.com

## Usage: .gitlab-ci.yml

FROM python:3.11-slim-bookworm
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
WORKDIR /projectOne

COPY requirements.txt /projectOne

RUN pip install --no-cache-dir django-bootstrap-v5 django-bootstrap5
RUN pip3 install --upgrade pip  && \
    pip3 install --no-cache-dir -r requirements.txt

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install -y libglib2.0-0 && \
    apt-get install -y graphviz

COPY . /projectOne 
ENV PORT 8000
ENTRYPOINT ["python3"] 
CMD ["manage.py", "runserver", "0.0.0.0:8000"]
