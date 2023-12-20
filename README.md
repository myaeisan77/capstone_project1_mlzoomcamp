# capstone_project1_mlzoomcamp

The problem is to forecast the sale of purchase item in each region.

**capstone_project_1.iynb file**  

Step 1 : Downloaded consumer behaviour dataset from kaggle ([https://www.kaggle.com/datasets/yasserh/wine-quality-dataset](https://www.kaggle.com/datasets/zeesolver/consumer-behavior-and-shopping-habits-dataset))  
Step 2 : Cleaned the datset and did EDA analysis (Purchase Amount and Location are closely related according to heatmap) 

Step 3 : Spilt the dataset into train, valid and test  
Step 4 : Trained Linear Regression model  
Step 5 : Find the least root mean square error (rmse) for the linear regression model with various r values 
Step 6 : All rmse value are the same and parameter tuning did not reduce the rmse.
Step 7 : Trained RandomForest Regression model and searched the best estimator and max_depth value to reduce the RMSE  

Finally it was found that Linear Regression model (rmse=0.44) performed slightly better than the RandomForest Regression model (rmse=0.45). So I built the model with linear regression model.

**train.py**  

Save the model.bin and dv.bin files from the .iynb notebook and trained again in train.py file  

**Dockerfile**  
Create dockerfile to setup environment for running predict.py file  




