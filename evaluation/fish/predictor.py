import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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

complete = f"{data_folder}/fish.csv"
learn = f"{data_folder}/fish.lrn.csv"
test = f"{data_folder}/fish.tes.csv"
prediction = f"{data_folder}/fish.pred.csv"
random_state=11776172

for folder in [data_folder, result_folder]:
    if not os.path.isdir(folder):
        print(f"Please be sure that the {folder} exists!")
        exit(1)
    

if not os.path.isfile(complete):
    print(f"Please be sure that {complete} is inside the data folder!")
    exit(1)
    

    


## Function to read csv files of dataset already configured for the files
def get_dataframe_from_file(path, seperator=","):
    return pd.read_csv(path, sep=seperator, header=0, encoding="utf-8", comment='#')

'''

PRE PROCESS
as full of strings and factors we decided to preprocess the whole file and replace the values with 
numbers and save it in a preprocessed state, for easier later handling.

'''

def preProcess(df):
    #Factors
    columns_to_factorize = ["Species"]
    for column in columns_to_factorize:
        df[column] = pd.Categorical(df[column])
        df[column] = df[column].cat.codes

    return df

def scale(df):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    columns = df.columns.to_list()
    columns.remove("Weight")
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

#generate new training and test file from df
split_into(df)
'''

EXTRACT LEARNING DATA

'''

## read in dataframe and get headers
df = get_dataframe_from_file(learn)

## create numpy array from values
numpy_array = df.values
target = numpy_array[:, [1]].flatten() # 1 --> weight
variables = np.delete(numpy_array, [1], axis=1) # 1 --> weight

'''

EXTRACT TEST DATA

'''

## read in dataframe and get headers
df = get_dataframe_from_file(test)

## create numpy array from values
numpy_array = df.values
test_target = numpy_array[:, [1]].flatten() # 1 --> weight
test_variables = np.delete(numpy_array, [1], axis=1) # 1 --> weight


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
    plt.suptitle(f'[Fish] Visualization of inhouse vs sklearn', fontsize=16)
    plt.title(information, fontsize=12)
    plt.ylabel("Predicted Weight")
    plt.xlabel("Annual Weight")
    plt.xticks(rotation=90)
    plt.legend(["Inhouse","Sklearn"])
    plt.axline([0, 0], [1, 1])
    return plt

def plot_scatter_gd():
    pred_gd = gd_inhouse(100000, 0.001, variables, target, test_variables)
    pred_gssk = gd_sklearn(1000, 0.001, variables, target, test_variables)
    plot = _plot_scatter(pred_gd, pred_gssk, test_target, "of Gradient Descent")
    plot.gcf().subplots_adjust(bottom=0.2, left=0.15)
    plot.savefig(f"{result_folder}/gd_fish_scatterplot", dpi=500)
    plot.close()

def plot_scatter_knn():
    pred_inhouse = knn_inhouse(3, "euclidean", variables, target, test_variables)
    pred_sklearn = knn_sklearn(3, "euclidean", variables, target, test_variables)
    plot = _plot_scatter(pred_inhouse, pred_sklearn, test_target, "of k nearest neighbours")
    plot.gcf().subplots_adjust(bottom=0.2, left=0.15)
    plot.savefig(f"{result_folder}/knn_fish_scatterplot", dpi=400)
    plot.close()

def predict_lasso_random():
    from utils import PerformanceMetrics
    pred_rf = random_forest(variables, target, test_variables)
    pred_ls = random_forest(variables, target, test_variables)

    rf_mse = PerformanceMetrics.mse(pred_rf, test_target)
    rf_mae = PerformanceMetrics.mae(pred_rf, test_target)
    rf_r2 = PerformanceMetrics.r2(pred_rf, test_target)

    ls_mse = PerformanceMetrics.mse(pred_rf, test_target)
    ls_mae = PerformanceMetrics.mae(pred_rf, test_target)
    ls_r2 = PerformanceMetrics.r2(pred_rf, test_target)
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
    with open(f"{result_folder}/knn_fish.csv", "w") as f:
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
    with open(f"{result_folder}/gd_fish.csv", "w") as f:
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


do()