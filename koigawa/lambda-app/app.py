import json
import pandas as pd
import altair as alt
from altair import datum
import numpy as np
import boto3
import os
import gtab
import urllib3
import requests
from datetime import date, timedelta
import pickle
import tensorflow
from tensorflow.keras.models import load_model
import shutil
from scipy.fftpack import fft, rfft

s3 = boto3.client('s3')

def handler(event, context):

    # Eventually change this to only run at POST call
    print(event)
    output = 'temp_val'
    status = 'temp_val'
    chart_spec = None

    if event['httpMethod'] == 'POST':
        key_word = json.loads(event['body'])["keyWord"]
        print("User value retrieved!")
        print(key_word)

        s3 = boto3.client('s3')
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('QueryTerms')

        item_key = {'input': key_word}
        response = table.get_item(Key=item_key)

        try:
            output = key_word + ", " + response['Item']['related_words']
            status = response['Item']['status']
        except:
            sagemaker_runtime = boto3.client('sagemaker-runtime')

            endpoint_name = os.getenv('SM_ENDPOINT_NAME')
            inference_component_name = os.getenv('SM_INFERENCE_COMPONENT_NAME')

            response = sagemaker_runtime.invoke_endpoint(
                    EndpointName = endpoint_name,
                    ContentType = "text/plain",
                    Body = key_word,
                    InferenceComponentName = inference_component_name
                    )

            print(response)
            response_list = (", ").join(response["Body"].read().decode('utf-8').split(";")[:3])

            item = {
                'input': key_word,
                'status': 'Cached',
                'related_words': response_list
                    }

            table.put_item(Item=item)
            status = 'New'
            output = key_word + ", " + response_list

        create_forecast_data(output)
        generate_predictions()
        generate_seasonal()

        df = pd.read_csv("s3://mads-siads699-capstone-cloud9/data/combined_forecast_df.csv")
        df.drop(columns=df.columns[0], axis=1, inplace=True)

        cols = [ col for col in df.columns if col not in ['date','is_prediction']]
        df = pd.melt(df,id_vars=["date","is_prediction"], value_vars=cols)

        df_fourier = pd.read_csv('s3://mads-siads699-capstone-cloud9/data/df_fourier.csv')
        df_fourier_dom_cycle = pd.read_csv('s3://mads-siads699-capstone-cloud9/data/df_fourier_dom_cycle.csv')

        selection = alt.selection_point(fields=['variable'],value=cols[0])
        opacity = alt.condition(selection, alt.value(1.0), alt.value(0.5))
        opacity_text = alt.condition(selection, alt.value(1.0), alt.value(0))

        chart = alt.Chart(df).mark_area().encode(
            x=alt.X('date:T', axis=alt.Axis(format="%Y-%b",labelAngle=-90),title=None),
            y=alt.Y("value:Q",
                    title="% of Total",
                                ).stack("normalize"),
            color=alt.Color('variable:N',title="Related terms",legend=alt.Legend(
                orient='none',
                legendX=170, legendY=-40,
                direction='horizontal',
                titleAnchor='middle')),
            opacity=opacity,
            order=alt.Order('sum(value):Q', sort='descending'),
            tooltip='variable:N'
        ).properties(
            width=470,
            height=180,
            title="Relative Popularity of Searched and Returned Terms"
        ).transform_filter(
            (datum.is_prediction == "N")
        ).add_params(
            selection
        )


        chart_two = alt.Chart(df).mark_line().encode(
            x=alt.X('date:T', axis=alt.Axis(format="%Y-%b",labelAngle=-90),title=None),
            y=alt.Y("value:Q", title="Gtab Value"),
            color=alt.Color('is_prediction:N',title="Is Forecast",legend=alt.Legend(
                orient='none',
                legendX=220, legendY=-40,
                direction='horizontal',
                titleAnchor='middle')),
            tooltip='is_prediction:N'
        ).properties(
            width=470,
            height=180,
            title="Forecasted Popularity for the Selected Term"
        ).transform_filter(
            selection
        )

        chart_three = alt.Chart(df_fourier).mark_line().encode(
            x=alt.X('date:T', axis=alt.Axis(format="%Y-%b",labelAngle=-90),title=None),
            y=alt.Y("value:Q", title="Amplitude"),
            color=alt.Color('variable:N',title="Terms",legend=None)
        ).properties(
            width=500,
            height=180,
            title="Seasonality for Selected Term Identified by Fourier Transform"
        ).transform_filter(
            selection
        )

        chart_four = alt.Chart(df_fourier_dom_cycle).mark_text(align="left",
            baseline="middle",size=20).encode(
            text="text",
            opacity=opacity_text,
        )

        with alt.themes.enable('fivethirtyeight'):
            chart_json = alt.vconcat(chart, chart_two,chart_three,chart_four).configure_axis(
                grid=False
                ).configure_axis(
                    labelFontSize=14,
                    titleFontSize=14
                ).configure_title(
                    fontSize=18,
                    orient='top',
                    anchor='middle'
                ).resolve_scale(
                    shape='independent',
                    color='independent'
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
        "body": json.dumps({'cache_status':status,'relevant_terms':output})
    }

def get_single_keyword_trend_data_gtab(keyword, region='US'):

    day_ago = date.today() - timedelta(days=1)
    year_ago = date.today() - timedelta(days=366)
    time_period = year_ago.strftime("%Y-%m-%d") + " " + day_ago.strftime("%Y-%m-%d")
    """
    Query Google Trends data using GTAB for a single keyword, region, and time period.

    Args:
        keyword (str): The keyword to search.
        region (str): Region code (default is 'US').
        time_period (str): Timeframe for the data (default is '2020-01-01 2024-10-11').

    Returns:
        pd.DataFrame: Google Trends data for the keyword with Date and Max Ratio (Interest) columns.
    """

    source_dir = '/var/lang/lib/python3.12/site-packages/gtab'
    destination_dir = '/tmp/gtab'

    # Create the destination directory if it doesn't already exist
    os.makedirs(destination_dir, exist_ok=True)

    print('The original path should be:',os.path.dirname(gtab.__file__))

    # Copy all content from source_dir to destination_dir
    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)

    t = gtab.GTAB(destination_dir)

    try:
        # Ensure keyword is a non-empty string
        if not keyword or not isinstance(keyword, str):
            raise ValueError("Invalid keyword provided.")

        print("Querying keyword: %s in region: %s for the period %s" % (keyword, region, time_period))


        # Query Google Trends using GTAB
        t.set_options(pytrends_config={"timeframe": time_period, 'geo': region})
        trend_data = t.new_query(keyword)

      # Check if data is empty
        if trend_data.empty:
            print("No data found for keyword: %s" % keyword)
            return None

        # Rename 'max_ratio' to the keyword
        trend_data = trend_data[['max_ratio']].rename(columns={'max_ratio': keyword})

        # Reset the index to move 'date' from the index to a column
        trend_data = trend_data.reset_index()

        return trend_data

    except Exception as e:
        print("An error occurred: %s" % e)
        return None

def create_forecast_data(keywords,region="US"):

    # Process the keywords and region
    keywords = [kw.strip() for kw in keywords.split(',') if kw.strip()]  # Clean whitespace and ensure each is a string
    region = region.strip()

    # Initialize an empty DataFrame to hold combined results
    combined_trend_data = pd.DataFrame()

    # Loop through each keyword and merge the results
    for keyword in keywords:
        trend_data = get_single_keyword_trend_data_gtab(keyword, region)

        if trend_data is not None:
            if combined_trend_data.empty:
                combined_trend_data = trend_data
            else:
                # Merge the current keyword's data into the combined DataFrame on 'date'
                combined_trend_data = pd.merge(combined_trend_data, trend_data, on='date', how='outer')

    combined_trend_data.to_csv('s3://mads-siads699-capstone-cloud9/data/combined_trend_data.csv', index=False)

def predict_future(model, keyword_data, time_step, num_predictions):
    # Get the last known data for the keyword
    last_sequence = keyword_data[-time_step:]  # The last time-step sequence from the data
    predictions = []

    for _ in range(num_predictions):
        # Reshape to match the input shape for the RNN
        predicted = model.predict(last_sequence.reshape(1, time_step, 1))
        predictions.append(predicted[0, 0])  # Get the predicted value

        # Update the sequence with the predicted data for the next step
        last_sequence = np.append(last_sequence[1:], predicted)  # Update the last sequence

    return predictions


# Function to generate future dates based on the last date in the dataframe
def generate_future_dates(last_date, num_predictions, interval_days=7):
    return pd.date_range(start=last_date + pd.Timedelta(days=interval_days), periods=num_predictions, freq=f'{interval_days}D')

def predict_future(model, keyword_data, time_step, num_predictions):
    # Get the last known data for the keyword
    last_sequence = keyword_data[-time_step:]  # The last time-step sequence from the data
    predictions = []

    for _ in range(num_predictions):
        # Reshape to match the input shape for the RNN
        predicted = model.predict(last_sequence.reshape(1, time_step, 1))
        predictions.append(predicted[0, 0])  # Get the predicted value

        # Update the sequence with the predicted data for the next step
        last_sequence = np.append(last_sequence[1:], predicted)  # Update the last sequence

    return predictions


# Function to generate future dates based on the last date in the dataframe
def generate_future_dates(last_date, num_predictions, interval_days=7):
    return pd.date_range(start=last_date + pd.Timedelta(days=interval_days), periods=num_predictions, freq=f'{interval_days}D')


def generate_predictions():
    # Import the output of DataExtractionGTAB.py
    data = pd.read_csv('s3://mads-siads699-capstone-cloud9/data/combined_trend_data.csv')
    # Import model
    bucket_name = 'mads-siads699-capstone-cloud9'
    # model_key = 'model/rnn_model_limited_data_v1.pkl'
    # model_path = '/tmp/rnn_model_limited_data_v1.pkl'
    model_key = 'model/LSTM40Epochs.keras'
    model_path = '/tmp/LSTM40Epochs.keras'
    s3.download_file(bucket_name, model_key, model_path)

    with open(model_path, 'rb') as file:
        model = load_model(file)
        #model = pickle.load(file)

    # variables
    num_predictions = 10
    time_step = 10 # Define time step (number of previous observations) considered - this needs to match model trainings

    # Extend df to have 10 future dates
    data['date'] = pd.to_datetime(data['date'])

    # Generate new dates with a 7-day interval, starting from the last date in df
    last_date = data['date'].iloc[-1]
    new_dates = generate_future_dates(last_date, num_predictions)

    predicted_dfs = [] # establish a list to store seperate df's per keyword

    # Run predictions on each keyword
    for keyword in data.columns[1:]: # loop through each unique keyword

        keyword_column_index = data.columns.get_loc(keyword)  # Get the column index for the keyword
        keyword_data = data[keyword].values  # Extract the data for the keyword
        future_predictions = predict_future(model, keyword_data, time_step, num_predictions=10)

        # add dataframe with future predictions for keyword to list
        predicted_dfs.append( pd.DataFrame({'date': new_dates, keyword: future_predictions}))

    # Merge DataFrames on the 'date' column
    merged_predictions_df = pd.concat(predicted_dfs, axis=1)

    # Remove duplicate 'date' columns (optional, as they will have the same values)
    merged_predictions_df = merged_predictions_df.loc[:, ~merged_predictions_df.columns.duplicated()]

    # Reset index to have a clean DataFrame
    merged_predictions_df.reset_index(drop=True, inplace=True)

    # Define the file path where the CSV will be saved (within the same directory)

    combined_trend_data_df = pd.read_csv("s3://mads-siads699-capstone-cloud9/data/combined_trend_data.csv")
    combined_trend_data_df["is_prediction"] = "N"
    merged_predictions_df["is_prediction"] = "Y"

    combined_df = pd.concat([combined_trend_data_df,merged_predictions_df])

    combined_df.to_csv('s3://mads-siads699-capstone-cloud9/data/combined_forecast_df.csv')

def generate_seasonal():
    data = pd.read_csv('s3://mads-siads699-capstone-cloud9/data/combined_trend_data.csv')

    df_fourier = pd.DataFrame()
    df_fourier_dom_cycle = pd.DataFrame()

    for word in data.columns[1:] :
        # removing the mean from the data to minimize distortion in the lower frequencies 
        X = (data[word] - data[word].mean()).values

        dates = pd.to_datetime(data['date'])

        # Compute the FFT
        fft_result = np.fft.fft(X)
        freqs = np.fft.fftfreq(len(fft_result), 1)

        # Identify the positive frequencies and their magnitudes
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = np.abs(fft_result[:len(fft_result)//2])

        # Find the index of the highest magnitude frequency
        max_index = np.argmax(positive_magnitude)
        dominant_freq = positive_freqs[max_index]

        # Create a mask to isolate the frequency with the highest magnitude
        mask = np.zeros_like(fft_result, dtype=complex)
        mask[max_index] = fft_result[max_index]
        mask[-max_index] = fft_result[-max_index]  # Include the symmetric part for real signals

        # Perform IFFT to reconstruct the signal for the highest magnitude frequency
        reconstructed_signal = np.fft.ifft(mask).real

        temp_df = pd.DataFrame({word:reconstructed_signal})

        df_fourier = pd.concat([df_fourier,temp_df], axis=1)

        # # Check if the frequency is non-zero
        if dominant_freq > 0:
            # Calculate the period in weeks (1/frequency)
            period_in_weeks = 1 / dominant_freq

            # Convert weeks to months (approximately 4.345 weeks per month)
            period_in_months = period_in_weeks / 4.345

            # Print human-readable cycle description
            if period_in_weeks < 4:
                temp_df_dom_cycle = pd.DataFrame({"variable":[word],
                                                  "value":[period_in_weeks],
                                                  "period":["weeks"]})
                df_fourier_dom_cycle = pd.concat([df_fourier_dom_cycle,temp_df_dom_cycle])

            else:
                temp_df_dom_cycle = pd.DataFrame({"variable":[word],
                                                  "value":[period_in_months],
                                                  "period":["months"]})
                df_fourier_dom_cycle = pd.concat([df_fourier_dom_cycle,temp_df_dom_cycle])

        else:
            print("The frequency is zero or invalid; no cycle can be determined.")


    # Additional processing for Altair

    df_fourier["date"] = data["date"].values
    cols = [ col for col in df_fourier.columns if col not in ['date']]
    df_fourier = pd.melt(df_fourier,id_vars=["date"], value_vars=cols)
    df_fourier_dom_cycle["text"] = df_fourier_dom_cycle.apply(lambda x: "The dominant cycle occurs every " + str(round(x["value"],2)) + " " + x["period"],axis=1)

    # Save as dataframe for later access
    df_fourier.to_csv('s3://mads-siads699-capstone-cloud9/data/df_fourier.csv')
    df_fourier_dom_cycle.to_csv('s3://mads-siads699-capstone-cloud9/data/df_fourier_dom_cycle.csv')