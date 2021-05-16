# Udacity_DataScience_Project2

**Table of Contents**

  1.	Installation
  2.	Project Motivation
  3.	File Descriptions
  4.	Results
  5.	Licensing, Authors, and Acknowledgements


**Installation**

Anaconda Navigator V.1.10.0 was utilised, with the code being written using Jupyter Notebook V.6.1.4. Python version 3 was used. Codes were then added to process_data.py and train_classifier.py that were provided by Udacity. Run.py was also provided by Udacity. Packages that are required for python include: sys Pandas, Numpy, sqlalchemy, nltk, sklearn and Pickle. Json, Flask and plotly were used for the visualisations.


**Project Motivation**

This Project is part of Lesson 3 of the Udacity Nanodegree for Data Science, where the aim is to create a machine learning pipeline that cleans, tidies and then categorises messages allocating a disaster response. The trained model was then tested and the results presented using Flask.
Within the project there were 3 aims:

1.	Write a program that loads the data from two csv files, cleans the data and then prepares it for machine learning use. The compiled data is then saved in an SQL database.

2.	In the second program, the aim is to create a tuned machine learning pipeline using GridSearch. The program is modular, with functions carrying-out specific tasks. Tasks included importing the data, creating a unique tokenisation function, creating the machine learning pipeline to tokenise, calculate tf-idf and then apply a classifier. RandomForest was chosen. 

3.	The final part of the project was to publish the results to an App using Flask and to create visualisations of the training data.


 **File Descriptions**
 
There are 3 modular python scripts that are required to be ran within this project. There are also 2 provided .csv files that contain the disaster messages that is used for training along with the categories of response.

Process_data.py – is a modular python program that loads, cleans and tidies the data for use within the analysis section. The final data is saved to an SQL database for use in the machine learning pipeline.

Train_classifier.py – is a modular python program that loads the data from an SQL database, splits the data the test and train, then applies a machine learning algorithm. A random forest was utilised, with GridSearch being used to optimise the best number of tress to create within the forest. A Classification result is created to present the Precision and F1 scores. The best model was saved to a pickle file.

Run.py – is a program that utilises Flask to present the classification of a message using the model. Visuals are also presented of the training data. Within the code, reference is made to the SQL database containing the training data along within the classifier.pkl file that contains the saved model. These would need to be updated for a future database and new model. 

Messages.csv – is the csv file containing the text messages received. The file also contains a unique ID for each message, the original non translated message as well as a field representing a the genre of the message (News, social or Direct). There are 26,248 messages in total 

Categories.csv- The categories file contains two fields – a unique id which is used to join to the messages, and the category of the response for the id. The category field is a string containing multiple responses, each split by a semi colon. 


**Results**

Within the Udacity IDE Terminal the results can be obtain by running the following commands within the root directory of the Udacity workspace.

To set up the database and run the ETL pipeline to clean the new data, the following should be ran:

python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

To run the Machine Learning pipeline and train and save a new classifier model the following should be ran:

python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

The Flask app can then be created by running the below command from within the app directory of the terminal

python run.py, with the app then being visable at: http://0.0.0.0:3001/

Within the train_classifier program, a unique tokenisation function was applied to the messages that removed stop words, applied lemmatisation as well as removing punctuation. TF-IDF was calculated as well. Train_test_split as utilised to separate the data in to training sets and tests sets. 20% of the data was reserved for the test set with the rest being used to train the model. A Random Forest classifier was utilised providing a high degree of accuracy (0.8), though this is variable across each of the multi-class responses. It was found that categories such as aid_related have a precision of 0.75 whilst aid_centers and hospitals had an average precision of 0.98. Further work would need to be undertaken to understand this discrepancy. 

Within the Flask App, 2 visualisations of the training data were requested to meet the criteria of the Rubric. Here, a bar chart of the Genre variable was presented along with the percent distribution of category of response. Within the genre summary there are three categories: direct, news and social, taking up 10.76k , 13k and 2,396 of the messages, respectively. Of all messages received (26k), the Direct category is represented in 76% of the messages, followed by Aid_related at 41% and weather_related at 27.8% 

One possible improvement for this model would be to over sample the classes where the response. This would restrict the model’s tendency to predict the classes that are better populated. Other classifiers were also utilised, such as KNN (K-Nearest Neighbours) but were not found to provide as high an accuracy level within category groups. 

**Licensing, Authors, Acknowledgements**

With thanks to Udacity for providing the templates and code structure for run.py, process.py and train_classifier.py
Thanks also to Figure Eight who provided messages.csv and categories.csv via Udacity
