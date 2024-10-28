import pandas as pd   
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.models import load_model



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


if __name__ == "__main__":

    # Import the output of DataExtractionGTAB.py
    data = pd.read_csv('combined_keyword_data.csv')
    # Import model
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
   
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

    # Get the path of the current script's folder
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Define the file path where the CSV will be saved (within the same directory)
    file_path = os.path.join(script_directory, 'predicted_keyword_data.csv')
    merged_predictions_df.to_csv(file_path, index=False)
    