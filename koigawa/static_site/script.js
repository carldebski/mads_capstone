/*
This is the JavaScrip that sends OPTION and POST API calls to API Gateway on AWS.
Once the results come back, they are rendered on HTML.

Generative AI in the form of ChatGPT was used to research on the right syntax and code snippets to use at some sections.
*/

function processInput() {
    // Get the user input value
    const userInput = document.getElementById("user-input").value;
    const messageValue = document.getElementById("entry-status");
    const apiUrl = 'https://kwwrgpxiqh.execute-api.us-east-1.amazonaws.com/websiteAPI';
    messageValue.textContent = "Thank you for the input! It can take a while for the results to appear. Now, if you encounter any issues, please refer to FAQ at the bottom of the page.";

    // Display the modified value in the output paragraph
    fetch(apiUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ "keyWord": userInput})  // Sending input as JSON
    })
    .then(response => response.json())  // Parse the JSON response
    .then(data => {
        const chartSpec = "altair/chart.json";
        document.getElementById("cache-status").innerHTML = "Cache Status: " + data.cache_status;
        document.getElementById("relevant-terms").innerHTML = "Relevant Terms: " + data.relevant_terms;
        document.getElementById("nlp-status").innerHTML = "NLP Status: " + data.nlp_status;
        document.getElementById("forecast-status").innerHTML = "Forecasting Status: " + data.forecast_status;
        // Use vegaEmbed to render the JSON chart
        vegaEmbed('#chart', chartSpec)
          .then(result => {
            // Handle successful rendering
            console.log("Chart successfully rendered!");
            document.getElementById("chart-status").innerHTML = "Chart rendered successfully!";
          })
          .catch(error => {
            console.error('Error rendering the chart:', error);
            document.getElementById("chart-status").innerHTML = "Failed to render the chart!";
          });
      })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById("output").textContent = 'Error calling API';
    });
}


