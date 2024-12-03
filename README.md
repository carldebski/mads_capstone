# Creating a cost-effective, lightweight tool hosted on Cloud services that forecasts popularity of terms on Google Trends for small retail business owners

<b>Last updated</b>: 2024/12/09<br>
<b>Members</b>: Jennifer Shumway, Carl Debski, Kento Oigawa


## Project Description
![screenshot](tool_screenshot.png)
<br><br>
This is a README for a repository that stores codes used on SIADS 699 Capstone project for Master of Applied Data Science program at the University of Michigan School of Information. 
As this was our final course for the graduate degree program, our main motivations were:
<ul>
<li>Apply data science knowledge that we have learnt throughout the program, in areas such as machine learning, data pipelines, data visualizations, natural language processing, data mining and cloud computing</li>
<li>Create some type of tangible output that will help solve and addresses some type of realistic business problems and needs</li>
</ul>
Based on the above motivation, we have decided to create a light-weight, cost-effective tool hosted in the cloud that performs forecasting of popularity of semantically related terms. To do so we leverage Google Trends, a website by Google that analyzes popularity of top search queries in Google Search across various regions and languages. Our intended audience for this tool were small retail business owners who can use the tool to keep up with both long-term and short-term trends of product or product types in order to make decisions about stocking and inventory purchases.
<br><br>
Due to time and resource constraints, we have determined minimally viable product (MVP) criteria as follows:

| Component  | MVP Criteria |
| ------------- | ------------- |
| Natural Language Processing (NLP)  | <ul><li>- [x] Take user search terms and come up with semantically related terms</li><ul>  |
| Forecasting and Seasonality  | <ul><li>- [x] Take the results of NLP processing and perform a forecast of popularity based on Google Trends data</li><li>- [x] Identify seasonal and non-seasonal trends in popularity of the terms</li><ul>|
| Cloud-based Architecture  | <ul><li>- [x] Integrate outputs from two ML solutions and deliver visualizations/dashboard to the user with the service being hosted on a static or dynamic website on AWS</li><li>- [x] Spend less than 200 USD to develop a solution that is able to run well beyond the end of the capstone project</li><ul>|

## Content
There are three main files within the repository under the acronym of each team member: 
<ul>
<li>“<b>cdebski</b>” contains the files for natural language processing (NLP). This includes the primary script, <b>get_related_words.py</b>, for returning the related search words, an exploratory Jupyter notebook, <b>related_terms.ipynb</b>, for generating specific values and visualizations for the report, and <b>requirements_cdebski.txt</b> file needed to set up the NLP environment. Embedding models are also saved here when generated, however, they are not pushed to Git due to its file size restrictions.</li>
<li>“<b>jshumway</b>” contains scripts, Jupyter notebooks and requirements.txt file for extracting Google Trends data and for performing both forecasting and identification of seasonality on that data.</li>
<li>“<b>koigawa</b>” contains codes, scripts,  and requirements.txt file used to build a static website and SageMaker application that incorporates scripts within “cdebski” and “jshumway” folders.</li>
</ul>

## Understanding Code & Example Workflows

### Natural Language Processing
The NLP function uses a three step process. 
<ul>
<li><b>Finding Relevant Terms Used:</b> Using a gensim word embedding model (fasttext-wiki-news-subwords-300) we are returning the terms with the highest cosine similarity to the search term.</li>
<li><b>Eliminating Spelling Variations:</b> Uses the Levenshtein Distance to measure the similarity to the search term and removes matching words within a distance of 2 or less. </li>
<li><b>Group Sorting Relevant Topics:</b> Incorporates semantic groups as a part of the sorting to find the most relevant terms. This is achieved using a lexical database (WordNet) by calculating the path distance to the search term to prioritize relevance and semantic group using the shortest distance.</li> 
</ul>

To perform all three steps in the NLP process:

Import the module.

        from get_related_words import get_similar_words

Call the function using the search term and optional number of returned results

        get_similar_words('Lemonade', n_words=5)

Results are returned in the format {search_word: [list_of_related_words]}

        {'lemonade: ['orangeade', 'limeade', 'beverage'...]}

Optionally, WordNet values can be retrieved or the path distance between words provided. 

        # import functions
        from get_related_words import get_hypernyms
        from get_related_words import get_hyponyms
        from get_related_words import get_synonyms
        from get_related_words import get_get_wordnet_path_similarity


        # retrieve hypernyms
        hypernyms = get_hypernyms('coffee')
        print('hypernyms:{}'.format(hypernyms))

        hypernyms: beverage


        # retrieve synonyms
        synonyms = get_synonyms('coffee')
        print('synonyms:{}'.format(synonyms))

        synonyms: [['java'], ['coffee_tree'], ['coffee_bean', 'coffee_berry']...]


        # retrieve hyponyms
        hyponyms = get_hyponyms('coffee')
        print('hyponyms:{}'.format(hyponyms))

        hyponyms: [['instant_coffee'], ['mocha', 'mocha_coffee'], ['drip_coffee']...]


        # return the path distance between two words in WordNet (the same word returns a 
        # distance of 1)
        term1 = 'beverage'
        term2 = 'mocha'
        dist = get_wordnet_path_similarity(term1, term2)
        print('The path distance between {} and {} is {}.'.format(term1, term2, dist))

        The path distance between beverage and mocha is 0.1
        

### Forecasting and Seasonality

In order to perform your own forecasting or duplicate the results we have illustrated it is recommended to take the following workflow:    

1. Data collection   
   a. __Data_extraction_gtab.ipynb__ contains the code necessary to extract data for any keyword for any region.
   b. All data we leveraged is available in the file __all_merged_data.csv__ and each of the following notebooks begin with importing this data and splitting it directly into train/dev/test partitions. 

3. Hyperparameter tuning and model training
   a. The final trained model is available as __lstm_40_epochs.keras__ alternatively you can follow our workflow through __hyper_parameter_tuning.ipynb__, __hyper_parameter_tuning_viz.ipynb__ and __model_training.ipynb__
   
5. Seasonality
   a. Lastly you can explore seasonality for any keyword via fourier transform with the notebook __fourier_transform_seasonality.ipynb__


Additionally, the scripts used in the application are available in the subfolder scripts and can be executed directly. 






## How to deploy the tool to AWS
In order to deploy the tool, you have to make sure that you have a valid AWS account and the correct configuration. Below are <b><ins>non-exhaustive</ins></b> instructions and details on configuration for the solution which was implemented for our project. Please note that there are many approaches to this and that there is no single solution. For example, instances can be much larger than the one we have used, depending on your specific needs and available resources.

### API Gateway 
<b>Why this service is needed</b>: This service acts as an entry point into AWS for the tool when you are passing values, as JSON payload, entered by the user. This service can be invoked by calling on the URL defined in the endpoint.<br><br>
<b>Instructions</b>: 
<ul>
<li>Create HTTP API</li>
<li>Configure Web API with POST and OPTIONS routes</li>
<li>Configure Cross-Origin Resource Sharing</li>
<li>Configure Integrations with Lambda function (Make sure you have a Lambda function made prior to this)</li>
<li>Take note of the url within “Default endpoint” and the name of Routes as this will be required to call API from JavaScript</li>
</ul>

### Elastic Container Repository (ECR)
<b>Why this service is needed</b>: In order for the tool to work, Lambda function and SageMaker endpoint both require Docker images. To store Docker images which can be deployed to the two services, the images must be pushed to ECR.<br>
<b>Instructions</b>:
<ul>
<li>Create a private repository within ECR that will contain two images: one for Lambda function and one for SageMaker</li>
</ul>

### EC2 
<b>Why this service is needed</b>: This is needed for deploying the tool by building and pushing codes as Docker image to Elastic Container Repository.<br><br>
<b>Configuration</b>:
<ul>
<li>Instance type: t2.micro</li>
<li>OS: Amazon Linux 2</li>
<li>EBS: gp2 of size 32 GiB</li>
</ul>
<b>Instructions</b>:
<ul>
<li>Launch Amazon Linux 2 instance</li>
<li>Install Python3.12, Docker</li>
<li>Depending on the storage size, increase volume if you encounter any storage issues when you are installing packages or building Docker images</li>
<li>Clone this github repository</li>
<li>Build and push content within koigawa/lambda_app as a Docker image into a repository you have created in Elastic Container Repository, where you can find scripts to run on EC2 are available there. Make sure that you start the Docker before doing so by running</li>
</ul>

        sudo systemctl start docker

<ul>
<li>Create DynamoDB instance (either via CLI or interface) and change value of “QueryTerms” into the instance name that you have chosen within handler function of koigawa/lambda_app/app.py</li>
</ul>

        table = dynamodb.Table('QueryTerms')
        item_key = {'input': key_word}
        response = table.get_item(Key=item_key)

### Lambda
<b>Why this service is needed</b>: This is the serverless tool which will house the main Python script for the tool. It is triggered by a POST call passed through by API Gateway and performs a bunch of tasks such as calling DynamoDB, calling SageMaker endpoint, and then saving JSON output from Altair charts rendered into S3.<br><br>
<b>Configuration</b>:
<ul>
<li>Architecture: x86_64</li>
<li>Memory: 1024 MB</li>
<li>Timeout: 15 mins</li>
<li>Environment variables:</li>
<ul>
<li>SM_ENDPOINT_NAME: <i>{Enter name of your endpoint here}</i></li>
<li>SM_INFERENCE_COMPONENT_NAME: <i>{Enter name of inference component name here}</i></li>
</ul>
</ul>
<b>Instructions</b>:
<ul>
<li>Create Lambda function from a Docker image that you have built in EC2 and sent to ECR</li>
<li>Configure IAM Role to ensure that the function can write log into CloudWatch Logs, run SageMaker and access S3</li>
<li>Ensure integration with API Gateway by configuring Lambda Integration within API Gateway</li>
</ul>

### S3 
<b>Why this service is needed</b>: Because the tool will be a static website, S3 will be used to store the main code (JavaScript, HTML and CSS) as well as any other relevant data such as JSON for the Altair chart which will get rendered to the user.<br><br>
<b>Configuration</b>:
<ul>
<li>Bucket Versioning: Enabled (Not required)</li>
<li>S3 static website hosting: Enabled</li>
<li>Block all public access: Disabled</li>
</ul>
<b>Instructions</b>:
<ul>
<li>Create an S3 with the above configuration</li> 
<li>Upload the content within koigawa/static_site into the newly created bucket</li>
<li>Click the URL link on Bucket website endpoint to confirm that it is working</li>
<li>On a local machine, download all files from <a href=https://huggingface.co/fse/fasttext-wiki-news-subwords-300/tree/main>this site</a> and, in the active directory, create model.tar.gz as per below code and save it as s3://{your_bucket_name}/model/model.tar.gz, as this is needed later in SageMaker setup</li>
</ul>

```
mkdir model_files
cp fast* model_files/
tar -czvf model.tar.gz -C model_files .
```

<ul>
<li>On a local machine, download jshumway/lstm_40_epochs.keras and save it as s3://{your_bucket_name}/model/lstm_40_epochs.keras.</li>
</ul>

### SageMaker
<b>Why this service is needed</b>: SageMaker is needed to host NLP process. The model will be called by Lambda function to have similar terms returned.<br><br>
<b>Configuration</b>:
<ul>
<li>Min CPU memory: 128</li>
<li>Min number of copies: 1</li>
<li>Instance type: ml.m5.large</li>
</ul>
<b>Instructions</b>:
<ul>
<li>Set up SageMaker studio</li>
<li>Within EC2, build and push Docker image from within the content of koigawa/nlp_sagemaker_app and note down the URI of the image</li>
<li>Register a model using URI of the image and the model saved within S3 bucket as s3://{your_bucket_name}/model/model.tar.gz</li>
<li>Deploy inference endpoint from the model using the configuration above</li>
</ul>

