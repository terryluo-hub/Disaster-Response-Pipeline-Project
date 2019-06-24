# Disaster Response Pipeline Project

# Table of Contents:

1. [Installation](#installation)
2. [Summary](#summary)
3. [Instruction](#instruction)


# Installation:

```
Files:

- app
| - template
| |- master.html 
| |- go.html  
|- run.py  

- data
|- disaster_categories.csv  
|- disaster_messages.csv  
|- process_data.py

- models
|- train_classifier.py 
```

### The following packages are required:

- json
- plotly
- pandas
- numpy
- sklearn
- sqlalchemy
- nltk
- flask

# Summary:

This project is about analyzing data regarding disaster reponse from Figure Eight. It uses natural language processing libary to classify messages based on historical data. The messages are from different resources. The webpage of the project gather the input of user message, then display the category of message it supposes to be in.


# Instruction:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



