# Mini-Machine Learning Projects

## [Stochastic Relative Strength Indicator Algo Trading]()
This model used StochasticRSI as the basis to time when to buy or sell a stock.
* This model used the Pandas, Numpy, and Matplotlib libraries.
* From the data given, a function that calculates the StochRSI was created and plotted onto the graph of the stock; from this new metric, we can get an estimate about whether a stock is either overbought or oversold. As a result, we can set thresholds that can tell us whether we should buy or sell this stock based on the newly calculated StochRSI. 
* UNDER CONSTRUCTION: A function showing how much money this strategy made in relation to simply holding the stock is an idea worth coming back to.
* Credit for the code used in this project is from the YouTube channel "Computer Science", with the title of the video being "Use Stochastic RSI And Python To Determine When To Buy And Sell Stocks"

## [Simple Moving Average Algo Trading]()
This model used Simple Moving Averages (SMA) as the basis to time when to buy or sell a stock.
* This model used the Pandas, Numpy, and Matplotlib libraries.
* Two SMA's were created (one short (30 days) and the other long (100 days)), with the assumption that each time these two averages crossed each other, that would be a signal to either buy or sell the asset/stock. 
* Both a buy and sell varaible were created in this project, and were plotted onto the graph as well.
* UNDER CONSTRUCTION: A function showing how much money this strategy made in relation to simply holding the stock is an idea worth coming back to.
* Credit for the code used in this project is from the YouTube channel "Computer Science", with the title of the video being "Algorithmic Trading Strategy Using Python"

# Neural Network Projects

## [Gem Stone Project](https://github.com/AndrewVandenberg/Portfolio/blob/main/TensorFlow_Gem_Project.ipynb)
A NN model that predicts the price of gemstones based on their features. 
* This model utilized Google Collab, along with the libraries pandas, numpy, matplotlib, skikitlearn, keras, and seaborn for this project.
* Data is from a Kaggle dataset.
* The accuracy of this model had a MAE of $4, with the mean of the data being around $500.
 
![Epochs vs Loss](https://github.com/AndrewVandenberg/Portfolio/blob/main/images/gem.png)

## [Traffic Sign Classification](https://github.com/AndrewVandenberg/Portfolio/blob/main/LeNet%20Traffic%20Sign%20Classification.ipynb)*
I used a LeNet-CNN to help better classify traffic signals.
* This project made use of matplotlib, seaborn, keras, sklearn, pandas, and numpy
* I utilized the Pickle library to...
* This project classified images to help self-driving cars better navigate on the roads autonomously.

<img src="https://github.com/AndrewVandenberg/Portfolio/blob/main/images/traffic.png" alt="Traffic Sign Predictions" width="500"/>

## [Car Sales Prediction](https://github.com/AndrewVandenberg/Portfolio/blob/main/Car%20Sales.ipynb)*
I utilized ANN's to predict the possible price range any given customer would be willing to consider based on features known about them.
* I used Jupyter Notebook for this project
* I utilized Tensorflow and Keras 

![Epochs vs Loss](https://github.com/AndrewVandenberg/Portfolio/blob/main/images/CarSales.png)

# Regression Projects

## [Boston Housing Pricing](https://github.com/AndrewVandenberg/Portfolio/blob/main/Boston%20Housing%20.ipynb)
I utilized linear regression to help predict the prices of houses in the boston area.
* I performed an EDA to discover what features were the most influential in predicting house prices.
* My MAE was 3.84, MSE 28.55, and RMSE 5.34

# Classification Projects

## [Titanic Project](https://github.com/AndrewVandenberg/Titanic)
I utilized logistic regression as a way to determine whether a passenger survived or not based on their various features.

* I utilized Jupyter Notebook, along with the libraries pandas, numpy, matplotlib, skikit learn, and seaborn for this project.
* I used the Kaggle data set Titanic to create a ML model.
* I used the logistic regression model to predict the survival of each passenger.
* The precision of this model came out to be around 78%.

![](https://github.com/AndrewVandenberg/Portfolio/blob/main/images/graph.png)

## [Iris Project](https://github.com/AndrewVandenberg/Portfolio/blob/main/Iris%20Project.ipynb)
This was a classification problem, which required me to classify different flowers into either one of three different species.
* I used K-Nearest Neighbor to solve this problem

## [Breast Cancer](https://github.com/AndrewVandenberg/Portfolio/blob/main/Breast%20Cancer.ipynb)
For this classification problem, I used Support Vector Machines to classify whether someone has breast cancer based on different features/variables.
* With this dataset, I was able to accurately predict whether someone has breast cancer 91% of the time!

## [Email Spam Filter](https://github.com/AndrewVandenberg/Portfolio/blob/main/Email%20Spam%20Filter.ipynb)*
This project used a Naive Bayes classifier to differentiate between spam and "ham" emails.
* With this model, I was able to achieve a 98% accuracy rate.

![Accuracy of Model](https://github.com/AndrewVandenberg/Portfolio/blob/main/images/email.png)

## [Movie Recommendation Model](https://github.com/AndrewVandenberg/Portfolio/blob/main/Movie%20Recommendations.ipynb)*
This model predicts potential movie interests to a viewer based on their reviews of previous movies.
* This model did not rely on the sklearn librbary or NN's; 
* Instead, **heavily** utilizing the pandas library.
* There is no ability to check the accuracy of this model.


# Forecasting Projects

## [Crime Rate in Chicago](https://github.com/AndrewVandenberg/Portfolio/blob/main/Crime%20Rate%20in%20Chicago.ipynb)*
This was my introduction to FaceBook's Prophet library. With this data set, I was able to produce predictions about future crime within the city of Chicago based on past, available, data.

* FB Prophet is used to forecast time-series data based on an additive model.
* I utilized Seaborn and Matplotlib to better understand the data visually.
* I was able to use the data to study the general trends within the data.

![](https://github.com/AndrewVandenberg/Portfolio/blob/main/images/crime.png)

Acknowledgements

\* These projects were taken from the course "Machine Learning Practical Workout: 8 Ral-World Projects," by Ryan Ahmed.
