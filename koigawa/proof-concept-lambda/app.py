import json
import pandas as pd
import altair as alt
from altair import datum
import numpy as np
import boto3

def handler(event, context):
    
    # Eventually change this to only run at POST call
    print(event)
    user_value = 'temp_val'
    output = 'temp_val'
    response_list = 'temp_val'
    chart_spec = None
    if event['httpMethod'] == 'POST':
        user_value = json.loads(event['body'])["userInput"]
        print("User value retrieved!")
        print(user_value)

        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('QueryTerms')
        item_key = {'input': user_value}

        response = table.get_item(Key=item_key)

        try:
            output = response['Item']['status']
        except:
            item = {
                'input': user_value,
                'status': 'Cached'
                    }
            table.put_item(Item=item)
            output = 'New'

        sagemaker_runtime = boto3.client('sagemaker-runtime')

        endpoint_name = 'web-sagemaker-nlp-1730723348977-Endpoint-20241104-122913'
        inference_component_name = 'web-sagemaker-nlp-1730723348977-20241104-1229130'

        response = sagemaker_runtime.invoke_endpoint(
                EndpointName = endpoint_name,
                ContentType = "text/plain",
                Body = user_value,
                InferenceComponentName = inference_component_name
                )

        response_list = response.split(";")
        
        df = pd.read_csv("s3://mads-siads699-capstone-cloud9/data/car_details.csv")
        df = df[['Make','Fuel Tank Capacity','Price','Year','Seating Capacity',"Model"]]

        user_value = 'BMW'
        selection = alt.selection_point(fields=['Year'])
        chart = alt.Chart(df).mark_circle().encode(
            x='Price:Q',
            y='Fuel Tank Capacity:Q',
            color=alt.Color('Model:N',legend=None),
            tooltip='Fuel Tank Capacity:N'
                ).properties(
                   width=500,
                   height=180
                ).properties(
                    title='Fuel Tank Capacity vs Price for '+ user_value
                ).add_params(
                    selection
                ).transform_filter(
                    (datum.Make == user_value)
                ).interactive()

        chart_two = alt.Chart(df).mark_bar().encode(
            x='Model:O',
            y='count(Year):Q',
            color=alt.Color('Model:N'),
            ).properties(
                width=500,
                height=180
            ).properties(
                title='Model Count for ' + user_value
            ).transform_filter(
                (datum.Make == user_value)
            ).transform_filter(
                selection
                )

        chart_json = alt.vconcat(chart, chart_two).configure_axis(
            grid=False
            ).to_json()
        s3 = boto3.client('s3')

        bucket_name = "mads-siads699-capstone-cloud9"
        file_name = "altair/chart.json"

        s3.put_object(
            Bucket=bucket_name,
            Key=file_name,
            Body=chart_json,
            ContentType='application/json'
        )

    
    return {
        'statusCode': 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST,OPTIONS",
            "Access-Control-Allow-Headers": "*"
        },
        "body": json.dumps({'cache_status':output,'relevant_terms':response_list})
        # "body": json.dumps({"message": user_value}) 
        # 'body': json.dumps('Hello from Lambda!')
    }

