# VisualAnalytics

We created this project for the exam "Visual Analytics" of the "Engineering in Computer Science" master degree.

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5 CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

Among all the available datasets on the web site we choose the one dealing with the “Heart Disease” (https://archive.ics.uci.edu/ml/datasets/heart+disease).The dataset contains 918 records with 12 attributes, selected by all published experiments over the 76 initial ones. 

After observing the dataset we thought about the creation of the visual representation oriented to medics. 
More specifically we made an .html file which will display the elaborated data in 3 different ways, such that the medic can observe specific behaviors and study them. Those 3 interactive representation are: a scatter plot, a parallel coordinates graph and a table which will display the info of the selected dots or lines in the graphs. 

Moreover it is also possible to use a machine learning python code based on the CatBoost classifier to predict the probability of a person to suffer in the future from an heart disease.

HOW TO USE THIS PROJECT:

1. Download the entire project
2. Open "visualization.html" with a web browser (preferably Mozilla Firefox) and use it!

For the machine learning part:

1. Run a terminal in the folder
2. type "python3 server.py" (you should also install all the necessary libraries)
3. Now it's possible to access the prediction page from the "visualization.html"
