FROM python:3.8
COPY requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt --index-url http://nexus.prod.uci.cu/repository/pypi-proxy/simple/ --trusted-host nexus.prod.uci.cu
