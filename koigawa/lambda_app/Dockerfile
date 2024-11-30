FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ./

RUN pip3  --no-cache-dir install -r requirements.txt --ignore-installed boto3 botocore

COPY app.py ./

CMD ["app.handler"]
