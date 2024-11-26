function processInput() {
    // Get the user input value
    const userInput = document.getElementById("userInput").value;
    const messageValue = document.getElementById("entry-status");
    const apiUrl = 'https://kwwrgpxiqh.execute-api.us-east-1.amazonaws.com/websiteAPI';
    messageValue.textContent = "Thank you for the input! It can take a while for the results to appear. In the meantime, maybe you should go refill your coffee :)";

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


