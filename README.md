# Credit_Defaulter_Analysis_and_Prediction
This project uses exploratory data analysis to identify patterns that can be used to detect potential credit defaulters. The data is used to identify the driving factors behind loan default and to assess the risk associated with lending to customers. The goal is to ensure that the applicants capable of repaying the loan are not rejected, while also minimizing the risk of loss for the company.

## Running the code

You will need Python 3.7 or higher to run this code. 

To run the code, first you will need to change the PATH variable in the file `database_main.py`, line 14, to the location of the folder where you have downloaded the dataset. 

Once that is done, you can run the notebook `EDA.ipynb` and the code should execute correctly.

Below is a outline of code run flow.

![image](https://user-images.githubusercontent.com/76738199/226205838-1cdb5c16-dab4-4f06-820f-c1938c46118f.png)

Data Files:

To give an overview of what each file does:
1) `mycredlib.py` - This file has all the necessary functions that are needed for the data to be cleaned, normalized and loaded into the database. We are using SQLite for this project. You can modify it and use any flavor of SQL of your choice.
2) `database_main.py` - This is the main file that reads the data files and uses the libraries defined in `mycredlib.py` file to load the data into database. 
3) `customer_segmentation.py` - This file uses previous_application.csv file to cluster the customer into distinct groups and use it as a feature to process the application.csv file.
4) `EDA.ipynb` - This is the only file that needs to be run manually. All the other dependecies will be imported in this file and run as needed. The data will be read from SQLite database and all EDA will be performed in this notebook file including charts. Finally, a few ML algorithms are run to predict the future customers if they will be defaulters or not.

Thank you for using our project! If you have any questions or issues, please do not hesitate to contact us.
