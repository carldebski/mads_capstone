import pandas as pd 
from scipy.fftpack import fft, rfft
from matplotlib import pyplot as plt 
import numpy as np
import matplotlib.dates as mdates

if __name__ == "__main__":

    # Import the output of DataExtractionGTAB.py
    data = pd.read_csv('combined_keyword_data.csv')
    
    # taking the keyword as a column name, date comes first 
    keyword = data.columns[1]
    # removing the mean from the data to minimize distortion in the lower frequencies 
    X = (data[keyword] - data[keyword].mean()).values

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

    # Scale the reconstructed signal proportionally based on the FFT magnitude
    #amplitude_factor = positive_magnitude[max_index]
    #reconstructed_signal *= amplitude_factor / np.max(positive_magnitude)

    # Plot the original signal and the highest magnitude frequency component with the date as x-axis
    plt.figure(figsize=(14, 6))
    plt.plot(dates, X, label="Original Data", linestyle='--', color='black')
    plt.plot(dates, reconstructed_signal, label="Seasonal Frequency (Freq: {:.2f})".format(dominant_freq), color='red')



    # Set the title and labels
    plt.title("Original Signal and Dominant Frequency Component")
    plt.xlabel("Date")
    plt.ylabel("Amplitude")

    # Format the x-axis to show only the year, with ticks every quarter (3 months)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonthday=15, interval=3))  # Tick every 3 months
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Show only the year
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())  # Optionally, add minor ticks for months

    # Rotate the x-axis labels for better visibility
    plt.xticks(rotation=45)

    # Display the legend
    plt.legend()

    # Show the plot
    plt.show()



    

    # Check if the frequency is non-zero
    if dominant_freq > 0:
        # Calculate the period in weeks (1/frequency)
        period_in_weeks = 1 / dominant_freq

        # Convert weeks to months (approximately 4.345 weeks per month)
        period_in_months = period_in_weeks / 4.345

        # Print human-readable cycle description
        if period_in_weeks < 4:
            print(f"The dominant cycle occurs every {period_in_weeks:.2f} weeks.")
        else:
            print(f"The dominant cycle occurs every {period_in_months:.2f} months.")
    else:
        print("The frequency is zero or invalid; no cycle can be determined.")
