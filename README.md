# Creating a cost-effective, lightweight tool hosted on Cloud services that forecasts popularity of terms on Google Trends for small retail business owners

<b>Date</b>: 2024/12/09<br>
<b>Members</b>: Jennifer Shumway, Carl Debski, Kento Oigawa


## Project Description
![screenshot](tool_screenshot.png)
<br>This is a README for codes used on a capstone project for Master of Applied Data Science at the University of Michigan School of Information. In our project, we aimed to create a cost-effective, cloud-based (AWS) tool that performs forecasting of semantically related terms on Google Trends. The motivation for the project was to apply data science knowledge we had learnt throughout the graduate program, and chose to make a cloud-based tool to implement user interface as in business environment there are many pre-built, ready-to-use tools and services available, meaning that there is increased need to be able to put things together in a timely manner and below budget.

## Content
There are three main files within the repository under the acronym of each team member: 
<ul>
<li>“<b>cdebski</b>” contains scripts, Jupyter notebooks and requirements.txt file for natural language processing (NLP).</li>
<li>“<b>jshumway</b>” contains scripts, Jupyter notebooks and requirements.txt file for extracting Google Trends data and for performing both forecasting and identification of seasonality on that data.</li>
<li>“<b>koigawa</b>” contains codes, scripts,  and requirements.txt file used to build a static website and SageMaker application that incorporates scripts within “cdebski” and “jshumway” folders.</li>
</ul>

## Understanding Code & Example Workflows

## How to deploy the tool to AWS
In order to deploy the tool, you have to make sure that you have a valid AWS account and the correct configuration. Below are <b><ins>non-exhaustive</ins></b> instructions and details on configuration for the solution which was implemented for our project. Please note that there are many approaches to this and that there is no single solution. For example, instances can be much larger than the one we have used, depending on your specific needs and available resources.
