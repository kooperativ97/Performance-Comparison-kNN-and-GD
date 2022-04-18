# Performance Comparison of kNN and GD

[![DOI](https://zenodo.org/badge/6467636.svg)](https://zenodo.org/badge/latestdoi/6467636)
## About

This project aims to analyze the difference of two custom written regression algorithms against the state of the art implementation of [sklearn](https://scikit-learn.org/stable/). The two algorithms are k-nearest neighbors (KNN) and gradient descend (gd). Both are fully implemented in python. Key points to the comparison are accuracy and efficiency of the two algorithms but we will furthermore compare mse, mae, r2. 

## Requirements

To run these experiments you need **python 3** installed with the given modules provided in the `requirements.txt` file. To install all the required dependencies use pip

````
pip install -r requirements.txt
````

Furthermore you need to download the datasets and put them into the right place.

### Datasets

There are two datasets you need to download and move to the right place.

#### Fish

The fish data set is a data set of 7 features with 159 rows of data. It contains information of different fish species as their measured lengths of different aspects of their body, weight and  species. The file is 5.88 KB in size and published under GPL 2 License in CSV format.

Download the fish dataset via [Kaggle](https://www.kaggle.com/aungpyaeap/fish-market?select=Fish.csv) and save it as `fish.csv` into the `evaluations/fish/data/` folder.

Alternatively use the dataset published on [Zenodo](https://zenodo.org/record/6467551) which has added metadata (name and datatype information) comforming the [W3C's Model for Tabular Data and Metadata on the Web](https://www.w3.org/TR/tabular-data-model/#dfn-embedding-metadata).

##### Metadata of Fish

The list below gives the columns with a short description and datatype.

The Fish dataset has 7 columns: 
 * Species: the species name of fish (STRING)
 * Weight: the weight of fish in g (INTEGER)
 * Length1: vertical length in cm (DECIMAL)
 * Length2: diagonal length in cm (DECIMAL)
 * Length3: cross length in cm (DECIMAL)
 * Height: height in cm (DECIMAL)
 * Width: diagonal width in cm (DECIMAL)

#### Employee

The employee data set is a data set of 13 features and 9222 rows of data. It contains annual salary information including gross pay and overtime pay for all active, permanent employees of Montgomery County, MD paid in calendar year 2016. It is 1566 KB in size and was published under CC0 in ARFF format. 

Download the fish dataset via [OpenML](https://www.openml.org/d/42125) and save the dataset as `employee.arff ` into the `evaluations/employee/data/` folder.

Alternatively use the dataset published on [Zenodo](https://zenodo.org/record/6467551)

##### Metadata of Employee

The list below gives the columns with a short description and datatype.

The Employee dataset has 13 columns: 
 * full_name: the full name of the employee (STRING)
 * gender: the sex of the employee (CHAR, either "F" or "M")
 * current_annual_salary: the current annual salary of the employee (DECIMAL)
 * 2016_gross_pay_received: the gross pay received of the employee in 2016 (DECIMAL)
 * 2016_overtime_pay: the overtime pay received of the employee in 2016 (DECIMAL)
 * department: the short name of the department (STRING)
 * department_name: the full name of the department (STRING)
 * division: the full name of the division of the department (STRING)
 * assignment_category: the type of employment (STRING)
 * employee_position_title: the job position (STRING)
 * underfilled_job_title: the job position which they are considered for, but not qualified (STRING)
 * date_first_hired: the date when first hired (DATE, with format MM/DD/YYYY)
 * year_first_hired: the year when first hired (INTEGER)

## Run The Experiment

Make sure you installed all modules and put the datasets into the right folders, check the requirements above. 

The experiment works the same for both datasets, just the location of the python executable is different, the name `predictor.py` is the same. 

To run the experiment open a command line window and navigate to the desired dataset 

#### Fish

```
cd evaluation/fish
python predictor.py
```

#### Employee

```
cd evaluation/employee
python predictor.py
```

### Procedure

Right before evaluation, each dataset will be split into two parts (learn and test set) of variable size in CSV format. 
Furthermore, the employee dataset is converted from ARFF to CSV before this step resulting in followin new files:   

* evaluation/fish/fish.lrn.csv              (File size: 17 KB) 
* evaluation/fish/fish.tes.csv              (File size: 2 KB)
* evaluation/employee/employee.csv          (File size: 1 506 KB) 
* evaluation/employee/employee.lrn.csv      (File size: 1 247 KB)
* evaluation/employee/employee.tes.csv      (File size: 139 KB)

Then, both kNN and GD are initialized / trained with the training subset of each dataset ("lrn") and then evaluated against the test set ("tes").

### Results

The results of the experiment should be created in the `results` folder of each data set and should contain two csv files with naming: `{knn|gd}_{dataset}.csv` and two scatter plots named `{knn|gd}_{dataset}_scatterplot.png`
The first part of the file name indicated the used algorithm, the second part the dataset, the optional third part gives further information. 


 * gd_fish.csv File size: 3 KB 
 * knn_fish.csv File size: 5 KB 
 * gd_fish_scatterplot.png File size: 301 KB Dimensions: 2560 x 1920 pixels 
 * knn_fish_scatterplot.png File size: 218 KB Dimensions: 2560 x 1920 pixels   
 
 * gd_employee.csv File size: 2,57 Bytes 
 * knn_employee.csv File size: 5,44 KB 
 * gd_employee_scatterplot.png File size: 794 KB Dimensions: 2560 x 1920 pixels 
 * knn_employee_scatterplot.png File size: 723 KB Dimensions: 2560 x 1920 pixels  

#### GD

The results from the gd algorithm is saved in a CSV file starting with `gd_` followed by the data set name employee or fish and the file type `.csv`. Below, all columns and its meaning is described shortly: 

* method: the method indicates whether the sklearn implementation or the self coded (inhouse) implementation was used. Possible values: {`inhouse`, `sklearn`}
* learning_rate: indicates the learning rate, a parameter of the GD algorithm it ranges from `0.01` to `0.0001`
* nepoch: indicates the number of epochs, another paramter of the GD algorithm. It ranges from `1 ` to `10 000`
* mse: a metric to indicate the mean squared error, the smaller the better, a positive double value. 
* mae: a metric to indicate the mean absolute error, the smaller the better, a positive double value.
* r_two: a metric to indicate the variance, a double value. 
* total_seconds: the seconds needed to run the algorithm with the given parameters, a positive double.

#### kNN

The results from the kNN algorithm is saved in a CSV file starting with `knn_` followed by the data set name employee or fish and the file type `.csv`. Below, all columns and its meaning is described shortly: 

* method: the method indicates whether the sklearn implementation or the self coded (inhouse) implementation was used. Possible values: {`inhouse`, `sklearn`}
* distanceMetric: indicates the distance metric, a parameter of the kNN algorithm.  Possible values: {`euclidean`, `manhattan`}
* neighbours: indicates the amount of neighbours, a parameter of the kNN algorithm. Ranges from `1` to `100`, a positive integer.
* mse: a metric to indicate the mean squared error, the smaller the better, a positive double value. 
* mae: a metric to indicate the mean absolute error, the smaller the better, a positive double value.
* r_two: a metric to indicate the variance, a double value. 
* total_seconds: the seconds needed to run the algorithm with the given parameters, a positive double.

> **Understanding the metrics and parameters**
>
> In order to better understand the topic, metrics and parameters please take a look at the following resources.
>
> * https://scikit-learn.org/stable/modules/model_evaluation.html
> * https://scikit-learn.org/stable/modules/sgd.html
> * https://scikit-learn.org/stable/modules/neighbors.html

## Generating the Plots

In order to generate the additional plots, you need to run the experiment at least once. Copy the `csv` files from the results folder into `plotting/evaluations` keeping their original name. The folder should contain 4 files namely: 

* gd_fish.csv
* gd_employee.csv
* knn_fish.csv
* knn_employee.csv

To copy the files from their folder into the plotting folder you can use these two commands:

```
cp evaluation/fish/result/*.csv plotting/evaluations/
cp evaluation/employee/result/*.csv plotting/evaluations/
```

Then open a command line prompt and navigate into the plotting folder

``` 
cd plotting
```

and execute the python file via

```
python plotter.py
```

The process may take several minutes depending on the specifications of the machine running the code. The script should yield 34 plots in the plots subfolder. All plots have unique names and follow the naming convention: 

`{algorithm}_{dataset}_{value of altered variable}_{metric}.png`

The other 4 plots are performance plots and are named: 

`{algorithm}_{dataset}_performance.png`


### Metadata 

All CSV files produced in this project have the following metadata fields as comments in the first rows: 
 * 	created (YYYY-MM-DD-HH-mm-SS)
 * 	column names (comma seperated)
 * 	datatypes (comma seperated)
 * 	random state (integer, only appended when applicable)

The metadata attribution is compliant to the [W3C's Model for Tabular Data and Metadata on the Web](https://www.w3.org/TR/tabular-data-model/#dfn-embedding-metadata)
