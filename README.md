# NFL_Yards_Thrown_Predictions_ANN
Predicted the yards thrown in a game of NFL using Machine Learning Models and also the Artificial Neural Networks.
The open source datasets are available on https://www.pro-football-reference.com/ and can be downloaded to do the necessary analysis.
The data has a feature that has the description of the event in the game. The description is split into three different features while performing NLP on the data.
The three different features are the "Pass Complete", "From Player", "To Player". 
Assuming the fact that the pass is complete only when we have a value for the yards_thrown, we removed the data with no yards_thrown data.
The predictions were done using the Artificial Neural Networks, XGBoost Regressor, Random Forest Regressor, and the ADABoost Regressor. 
The predictions' output metrics were pretty close for the first three models but the ADABoost model failed to perform as well as the other models.
THANK YOU!
