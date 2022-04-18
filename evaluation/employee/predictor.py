import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import arff
import sys
import os.path
from datetime import datetime

sys.path.insert(1, '../../algorithms/knn')
sys.path.insert(1, '../../algorithms/gd')
sys.path.insert(1, '../../algorithms')

def do():
    save_nn_csv()
    plot_scatter_knn()
    save_gd_csv()
    plot_scatter_gd()
    
data_folder = "data"
result_folder = "results"

arff_file = f"{data_folder}/employee.arff"
complete = f"{data_folder}/employee.csv"
learn = f"{data_folder}/employee.lrn.csv"
test = f"{data_folder}/employee.tes.csv"
prediction = f"{data_folder}/employee.pred.csv"
random_state=11776172

for folder in [data_folder, result_folder]:
    if not os.path.isdir(folder):
        print(f"Please be sure that the {folder} exists!")
        exit(1)
   
if not os.path.isfile(arff_file):
    print(f"Please be sure that {arff_file} is inside the data folder!")
    exit(1)   
   
## Function to read csv files of dataset already configured for the files
def get_dataframe_from_file(path, seperator=","):
    return pd.read_csv(path, sep=seperator, header=0, encoding="utf-8", comment='#')

'''

ARFF

'''

def arff_to_csv():
    employees = arff.load(open(arff_file, 'r'))
    attributes = employees["attributes"]
    attr = [];
    for a in attributes:
        attr.append(a[0])
    employeedata = pd.DataFrame(employees["data"], columns = attr)
    with open(complete, "w") as f:
        print("#created {date}".format(date=datetime.today().strftime('%Y-%m-%d-%H:%M:%S')), file=f)
        print("#name {columns}".format(columns=','.join(employeedata.columns)), file=f)
        print("#datatype {types}".format(types=','.join([str(a) for a in employeedata.dtypes])), file=f)
        employeedata.to_csv(f, sep=",", encoding="utf-8", index=False)

#convert the arff to csv file 
arff_to_csv()


'''

PRE PROCESS
as full of strings and factors we decided to preprocess the whole file and replace the values with 
numbers and save it in a preprocessed state, for easier later handling.

'''
def gross_pay_placer(row):
    if pd.isna(row["2016_gross_pay_received"]) and row["year_first_hired"] == 2016:
        row["2016_gross_pay_received"] = row["current_annual_salary"] / 12

    return row

def preProcess(df):
    #Factors
    columns_to_factorize = ["division", "gender", "department", "assignment_category", "employee_position_title", "underfilled_job_title"]
    for column in columns_to_factorize:
        df[column] = pd.Categorical(df[column])
        df[column] = df[column].cat.codes

    # Datetime to unix timestamp
    df["date_first_hired"] = pd.to_datetime(df["date_first_hired"])
    df["date_first_hired"] = (df["date_first_hired"] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

    #Fill NAs in overtime pay as NAN is equal to no overtime payment equal to 0
    df["2016_overtime_pay"].fillna(value=0, inplace=True)

    #When employee is employed for less than a month, no gross_pay is set, so we set it to 1 months worth of salary
    df = df.apply(lambda row: gross_pay_placer(row), axis=1)

    #Drop department_name as department and dep_name are the same (just the full name)
    #Drop full name as it does not contain any usefull informatin (except for the name)
    df.drop(columns=["department_name", "full_name"], inplace=True)

    #One Row with missing gross pay still in the list, but employed since 2008, we drop the 1 row.
    df.dropna(inplace=True)

    return df



def scale(df): 
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    columns = df.columns.to_list()
    columns.remove("current_annual_salary")
    df[columns] = scaler.fit_transform(df[columns])
    return df

df = get_dataframe_from_file(complete)    
df = preProcess(df)
df = scale(df)
'''

SPLIT INTO TRAINING AND TEST DATA

'''
# Writes the complete data into test and training csv
def split_into(df):
    from sklearn.model_selection import train_test_split

    traindf, testdf = train_test_split(df, test_size=0.1, random_state=random_state) 
    
    if not os.path.isfile(test) or not os.path.isfile(learn): 
        with open(learn, "w") as f:
            print("#created {date}".format(date=datetime.today().strftime('%Y-%m-%d-%H:%M:%S')), file=f)
            print("#randomstate {state}".format(state=random_state), file=f)
            print("#name {columns}".format(columns=','.join(traindf.columns)), file=f)
            print("#datatype {types}".format(types=','.join([str(a) for a in traindf.dtypes])), file=f)
            traindf.to_csv(f, sep=",", encoding="utf-8", header=True, index=False)
        with open(test, "w") as f:
            print("#created {date}".format(date=datetime.today().strftime('%Y-%m-%d-%H:%M:%S')), file=f)
            print("#randomstate {state}".format(state=random_state), file=f)
            print("#name {columns}".format(columns=','.join(testdf.columns)), file=f)
            print("#datatype {types}".format(types=','.join([str(a) for a in testdf.dtypes])), file=f)
            testdf.to_csv(f, sep=",", encoding="utf-8", header=True, index=False)


# uncomment to generate new training and test file from df
split_into(df) 

'''

EXTRACT LEARNING DATA

'''

## read in dataframe and get headers 
df = get_dataframe_from_file(learn)

## create numpy array from values
numpy_array = df.values                     
target = numpy_array[:, [1]].flatten()  # 1 --> current_annual_salary 
variables = np.delete(numpy_array, [1], axis=1) # 1 --> current_annual_salary

'''

EXTRACT TEST DATA

'''

## read in dataframe and get headers 
df = get_dataframe_from_file(test)

## create numpy array from values
numpy_array = df.values 
test_target = numpy_array[:, [1]].flatten() # 1 --> current_annual_salary
test_variables = np.delete(numpy_array, [1], axis=1) # 1 --> current_annual_salary


'''

LEARN AND PREDICT

'''

def gd_inhouse(epoch, learning_rate, X, y, X_test):
    from GD import GD

    clf = GD(nepoch = epoch, learning_rate = learning_rate)
    clf.fit(X, y)
    return clf.predict(X_test)

def gd_sklearn(epoch, learning_rate, X, y, X_test):
    from sklearn.linear_model import SGDRegressor

    clf = SGDRegressor(max_iter=epoch, alpha=learning_rate)
    clf.fit(X, y)
    return clf.predict(X_test)

def knn_inhouse(k, distance_metric, X, y, X_test):
    from knn import KNN

    clf = KNN(k=k, distance_metric=distance_metric)
    clf.fit(X, y)
    return clf.predict(X_test)

def knn_sklearn(k, distance_metric, X, y, X_test):
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neighbors import DistanceMetric

    clf = KNeighborsRegressor(n_neighbors=k, metric=distance_metric)
    clf.fit(X, y)
    return clf.predict(X_test).tolist()


def _plot_scatter(inhouse, sklearn, yreal, information=""):
    #plot predicted to real values
    df = pd.DataFrame({"inhouse": inhouse, "sklearn": sklearn, "real": yreal})

    plt.scatter('real', 'inhouse', data=df, color="red", alpha=0.4)
    plt.scatter('real', 'sklearn', data=df, color="olive", alpha=0.4)

    #titles and descriptions
    plt.suptitle(f'[employee] Visualization of inhouse vs sklearn vs real', fontsize=16)
    plt.title(information, fontsize=12)
    plt.ylabel("Predicted Salary")
    plt.xlabel("Annual Salary")
    plt.xticks(rotation=90)
    plt.legend(["Inhouse","Sklearn"])
    plt.axline([0, 0], [1, 1])
    return plt

def plot_scatter_gd():
    pred_gd = gd_inhouse(100000, 0.001, variables, target, test_variables)
    pred_gssk = gd_sklearn(1000, 0.001, variables, target, test_variables)
    plot = _plot_scatter(pred_gd, pred_gssk, test_target, "of Gradient Descent")
    plot.gcf().subplots_adjust(bottom=0.2, left=0.15)
    plot.savefig(f"{result_folder}/gd_employee_scatterplot", dpi=500)
    plot.close()

def plot_scatter_knn():
    pred_inhouse = knn_inhouse(2, "euclidean", variables, target, test_variables)
    pred_sklearn = knn_sklearn(2, "euclidean", variables, target, test_variables)
    plot = _plot_scatter(pred_inhouse, pred_sklearn, test_target, "of k nearest neighbours")
    plot.gcf().subplots_adjust(bottom=0.2, left=0.15)
    plot.savefig(f"{result_folder}/knn_employee_scatterplot", dpi=500)
    plot.close()

def predict_lasso_random(): 
    from utils import PerformanceMetrics
    pred_rf = random_forest(variables, target, test_variables)
    pred_ls = lasso(variables, target, test_variables)

    rf_mse = PerformanceMetrics.mse(pred_rf, test_target)
    rf_mae = PerformanceMetrics.mae(pred_rf, test_target)
    rf_r2 = PerformanceMetrics.r2(pred_rf, test_target)

    ls_mse = PerformanceMetrics.mse(pred_ls, test_target)
    ls_mae = PerformanceMetrics.mae(pred_ls, test_target)
    ls_r2 = PerformanceMetrics.r2(pred_ls, test_target)
    print("Metrics for Random Forest")
    print(f"MSE: {rf_mse}")
    print(f"MAE: {rf_mae}")
    print(f"r2: {rf_r2}")

    print("Metrics for Lasso")
    print(f"MSE: {ls_mse}")
    print(f"MAE: {ls_mae}")
    print(f"r2: {ls_r2}")

def save_nn_csv():
    from utils import PerformanceMetrics
    with open(f"{result_folder}/knn_employee.csv", "w") as f:
        distanceMetric = ['euclidean', 'manhattan']
        neighbs = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,50,100]
        print("#created {date}".format(date=datetime.today().strftime('%Y-%m-%d-%H:%M:%S')), file=f)
        print("#name method,distanceMetric,neighbours,mse,mae,r_two,total_seconds", file=f)
        print("#datatype string,string,integer,double,double,double,double", file=f)
        print("method,distanceMetric,neighbours,mse,mae,r_two,total_seconds", file=f)
        for method, method_name in [(knn_inhouse, "inhouse"), (knn_sklearn, "sklearn")]:
            for dM in distanceMetric:
                for n in neighbs:
                    before = datetime.now()
                    pred = method(n, dM, variables, target, test_variables)
                    after = datetime.now()
                    delta = after - before
                    total_seconds = delta.total_seconds()
                    mse = PerformanceMetrics.mse(pred, test_target)
                    mae = PerformanceMetrics.mae(pred, test_target)
                    r2 = PerformanceMetrics.r2(pred, test_target)
                    print(f"{method_name},{dM},{n},{mse},{mae},{r2},{total_seconds}", file=f)
                    print(f"{method_name},{dM},{n},{mse},{mae},{r2},{total_seconds}")

def save_gd_csv():
    from utils import PerformanceMetrics
    with open(f"{result_folder}/gd_employee.csv", "w") as f:
        learning_rate = [0.01,0.001,0.0001]
        nepoch = [1,10,100,1000,10000]
        print("#created {date}".format(date=datetime.today().strftime('%Y-%m-%d-%H:%M:%S')), file=f)
        print("#name method,learning_rate,nepoch,mse,mae,r_two,total_seconds", file=f)
        print("#datatype string,double,integer,double,double,double,double", file=f)
        print("method,learning_rate,nepoch,mse,mae,r_two,total_seconds", file=f)
        for lr in learning_rate:
            for n in nepoch:
                before = datetime.now()
                pred = gd_inhouse(n, lr, variables, target, test_variables)
                after = datetime.now()
                delta = after - before
                total_seconds = delta.total_seconds()
                mse = PerformanceMetrics.mse(pred, test_target)
                mae = PerformanceMetrics.mae(pred, test_target)
                r2 = PerformanceMetrics.r2(pred, test_target)
                print(f"gd_inhouse,{lr},{n},{mse},{mae},{r2},{total_seconds}", file=f)
                print(f"gd_inhouse,{lr},{n},{mse},{mae},{r2},{total_seconds}")
        for lr in learning_rate:
            for n in nepoch:
                before = datetime.now()
                pred = gd_sklearn(n, lr, variables, target, test_variables)
                after = datetime.now()
                delta = after - before
                total_seconds = delta.total_seconds()
                mse = PerformanceMetrics.mse(pred, test_target)
                mae = PerformanceMetrics.mae(pred, test_target)
                r2 = PerformanceMetrics.r2(pred, test_target)
                print(f"gd_sklearn,{lr},{n},{mse},{mae},{r2},{total_seconds}", file=f)
                print(f"gd_sklearn,{lr},{n},{mse},{mae},{r2},{total_seconds}")


def knnCV(X, y, X_test, n_folds=10):
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import KFold
    from sklearn.model_selection import GridSearchCV

    knn = KNeighborsRegressor()

    tuned_parameters = [{'n_neighbors': [1,2,3,4,5,6,7,8,9,10,15,20,50,100], 'metric': ["euclidean", "manhattan"]}]

    clf = GridSearchCV(knn, tuned_parameters, cv=n_folds, refit=True)
    clf.fit(X, y)
    print("Best parameters: ")
    [print(f"{a}: {clf.best_params_[a]}") for a in clf.best_params_.keys()]
    pred = clf.predict(X_test)
    return (pred, clf.best_params_)

def doKnnCV():
    from utils import PerformanceMetrics
    pred, best_params = knnCV(variables, target, test_variables)

    print("MSE: {}".format(PerformanceMetrics.mse(pred, test_target)))
    print("MAE: {}".format(PerformanceMetrics.mae(pred, test_target)))
    print("R2: {}".format(PerformanceMetrics.r2(pred, test_target)))


do()