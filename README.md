# Project Documentation: Airline Ticket Price Prediction


## Table of Contents


**Introduction**

**Dataset Description**

**Project Objectives**

**Project Structure**

**Data Ingestion**

**Data Transformation**

**Model Training**

**Training Pipeline**

**Prediction Pipeline**

**Flask**

**Logging**

**Exception Handling**

**Utils**

**Conclusion**



## 1. Introduction


The project aims to predict airline ticket prices based on various features such as airline, source city, departure time, and more, using the "Airline Ticket Price Dataset." This document provides a comprehensive overview of the project, including its structure, processes, and supporting scripts.


## 2. Dataset Description

**Dataset Name**: Airline Ticket Price Dataset

**Description**: The dataset contains information on airline ticket prices and related features. The features include:

**Airline**: The airline operating the flight.

**Source City**: The city from which the flight departs.

**Departure Time**: The time of departure.

**Stops**: The number of stops in the journey.

**Arrival Time**: The time of arrival.

**Destination City**: The city where the flight lands.

**Class**: The class of travel (economy, business).

**Duration**: The total duration of the flight.

**Days Left**: The number of days left before the flight date.

**Price**: The price of the ticket.



## 3. Project Objectives


**Data Ingestion**: Load and explore the dataset.

**Data Transformation**: Clean, preprocess, and transform the dataset for modeling.

**Model Training**: Train various regression models to predict airline ticket prices.

**Pipeline Creation**: Develop a pipeline for data transformation, model training, and prediction.

**Supporting Scripts**: Provide scripts for setup, exception handling, utilities, and logging.



## 4. Project Structure

```
AirlineTicketPricePrediction/
│
├── artifacts/
│   ├── (best)model.pkl
│   ├── linearRegression.pkl
│   ├── Lasso.pkl
│   ├── Ridge.pkl
│   ├── ElasticNet.pkl
│   ├── DecisionTreeRegressor.pkl
│   ├── RandomForestRegressor.pkl
│   ├── GradientBoostingRegressor.pkl
│   ├── AdaBoostRegressor.pkl
│   ├── XGBoostRegressor.pkl
│   ├── raw.csv
│   └── preprocessor.pkl
│
├── notebooks/
│   ├── flight_prices.jpeg
│   ├── airline_ticket_prices.csv
│   ├── model_evaluation(chart).jpg
│   └── Airline_Ticket_Price_Prediction.ipynb
│
├── src/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_training.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
│
├── templates/
│   ├── index.html
│   └── results.html
│
├── static/
│   ├── plane (1).jpeg
│   └── style.css
│
├── app.py
├── .gitignore
├── requirements.txt
├── README.md
└── setup.py

```


## 5. Data Ingestion

The data ingestion file is used to ingest the data from notebook/data, split it into train, test, and raw CSV files, and save them into the artifacts folder.



## 6. Data Transformation

The data transformation file is used to perform exploratory data analysis (EDA), including encoding and preprocessing the data, and saving the encoded data.



## 7. Model Training

The model training file is used to train the model with various regression algorithms, saving the best model as a pickle file (.pkl) in the artifacts folder, and storing all models.
![download (2)](https://github.com/user-attachments/assets/84025eb0-fc50-4cf7-83a2-2d9024ff0173)


## 8. Training Pipeline

This file is used to run data ingestion, data transformation, and model training scripts in sequence.


## 9. Prediction Pipeline

This file is used to predict ticket prices using the best_model.pkl file and preprocess the data using preprocessor.pkl.


## 10. Static

**static/style.css**: Provides the theme for the index.html and results.html pages.


## 11. Templates

**templates/index.html**: Creates a form to get data input from the user.

**templates/results.html**: Displays the predicted results of the model.


## 12. Flask (app.py)

This file posts data from the form (index.html), predicts the results using prediction_pipeline.py, and displays the results.

![airplane_Ticket_Price_Prediction](https://github.com/user-attachments/assets/ba00f4ed-498a-43bf-823d-649e8e4b9873)

![airplane_ticket_price_Prediction_results](https://github.com/user-attachments/assets/4d168d1a-4d9c-4d1d-8e97-3f286e6e55df)


## 13. Logger

This file saves logs, recording the execution flow and errors.


## 14. Exception Handling

This file contains the exception handling code for errors, ensuring they are caught and logged appropriately.


## 15. Utils

This file contains utility functions used throughout the project for common tasks, such as creating directories.


## 16. Conclusion

This documentation provides a comprehensive overview of the "Airline Ticket Price Prediction" project, covering data ingestion, transformation, model training, pipeline creation, and supporting scripts. The provided code examples illustrate the implementation of various components, ensuring a robust and scalable project structure.
