### Example call from terminal 
# python DataExtractionGTAB.py --keywords "computer, phone, tablet" --region "US"



import pandas as pd   
import numpy as np
import gtab
import argparse
import os

def get_single_keyword_trend_data_gtab(keyword, region='US', time_period='2020-01-01 2024-10-11'):
    """
    Query Google Trends data using GTAB for a single keyword, region, and time period.
    
    Args:
        keyword (str): The keyword to search.
        region (str): Region code (default is 'US').
        time_period (str): Timeframe for the data (default is '2020-01-01 2024-10-11').
    
    Returns:
        pd.DataFrame: Google Trends data for the keyword with Date and Max Ratio (Interest) columns.
    """
    # Initialize GTAB object
    t = gtab.GTAB()

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



if __name__ == "__main__":

     # Set up argparse to handle command-line inputs
    parser = argparse.ArgumentParser(description="Query Google Trends data using GTAB.")
    parser.add_argument('-k', '--keywords', type=str, required=True, 
                        help="Comma-separated list of keywords to search for trends")
    parser.add_argument('-r', '--region', type=str, default='US', 
                        help="Region code (e.g., 'US-MI' for Michigan, 'US' for the entire USA)")
    
    # Parse arguments from the command line
    args = parser.parse_args()

    # Process the keywords and region
    keywords = [kw.strip() for kw in args.keywords.split(',') if kw.strip()]  # Clean whitespace and ensure each is a string
    region = args.region.strip()

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
    
   
    # Get the path of the current script's folder
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Define the file path where the CSV will be saved (within the same directory)
    file_path = os.path.join(script_directory, 'combined_keyword_data.csv')
    combined_trend_data.to_csv(file_path, index=False)