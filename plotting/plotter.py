import matplotlib.pyplot as plt
import pandas as pd
import random
import os 

DATA_SALARY = "employee"
DATA_FISH = "fish"

DATA_ALL = [DATA_SALARY, DATA_FISH]

TYPE_KNN = "knn"
TYPE_GD = "gd"

TYPE_ALL = [TYPE_KNN, TYPE_GD]

FOLDER_PLOTS = "plots"
FOLDER_EVALUATIONS = "evaluations"

for folder in [FOLDER_PLOTS, FOLDER_EVALUATIONS]:
    if not os.path.isdir(folder):
        print(f"Please be sure that the {folder} exists!")
        exit(1)

COLORS = ["saddlebrown","lawngreen", "plum", "deeppink",  "black", "indianred", "darkred", "coral", "moccasin", "darkseagreen", "mediumturquoise", "slategray", "indigo", "crimson", "olive"]
random.shuffle(COLORS)

## Function to read csv files of dataset already configured for the files
def get_dataframe_from_file(path, seperator=","):
    df = pd.read_csv(path, sep=seperator, header=0, encoding="utf-8", comment='#')
    return df 

# Function to get path for a specific dataset and type of evaluation
def get_path_of_set_for(dataset, typ):
    return f"{FOLDER_EVALUATIONS}/{typ}_{dataset}.csv"

# Function to get the path to save the plot for a specific dataset and type
def get_path_to_save_plot_for(dataset, typ, specific=""):
    specific = f"_{specific}" if len(specific) else ""
    return f"{FOLDER_PLOTS}/{typ}_{dataset}{specific}.png"

# Function to save figure as plot
def save_figure_as(plot, path, dpi=600, margin=0.23):
    plot.gcf().subplots_adjust(bottom=margin)
    plot.savefig(f"{path}", dpi=dpi)
    plot.close()
    plot.figure()
    
for TYPE in TYPE_ALL:
    for DS in DATA_ALL:
        path = get_path_of_set_for(DS, TYPE)
        if not os.path.isfile(path):
            print(f"Did not found {path}, aborting.")
            exit(1)

''' 

PLOTTING KNN

'''
def plot_knn_scores():
    for DS in DATA_ALL:
        df = get_dataframe_from_file(get_path_of_set_for(DS, TYPE_KNN))
        #DistanceMetric,Neighbours,mse,mae,r2,total_seconds
        df_headers = list(df.columns.values)
        print(df_headers)
        legend = []
        n = 0
        for dM in reversed(df["distanceMetric"].unique()):
            for metric in ["mse", "mae", "r_two"]:
                filt1 = df["distanceMetric"] == dM
                for method_name in df["method"].unique():
                    filt2 = df["method"] == method_name
                    plt.plot('neighbours', metric, data=df.loc[filt1 & filt2], color=COLORS[n], ls=['-','--','-.',':'][n%4] , linewidth=2, alpha=0.7)
                    legend.append(method_name)
                    n+=1
                
                plt.legend(legend, fontsize="medium")
                plt.ylabel(metric)
                plt.xlabel("Number of neighbors")
                plt.suptitle(f"[{DS}] KNN Evaluation of {metric} ", fontsize=16)
                plt.title("Comparing {}".format(", ".join(df["method"].unique())), fontsize=12)
                legend = []
                n = 0
                
                save_figure_as(plt, get_path_to_save_plot_for(DS, TYPE_KNN, f"{dM}_{metric}"))

        df.drop(columns=["neighbours", "mse", "mae", "r_two"], inplace=True)
        values = [v[0] for v in df.groupby(["method", "distanceMetric"]).mean().values]
        headers = ["inhouse\neuclidean", "inhouse\nmanhattan", "Sklearn\neuclidean", "sklearn\nmanhattan"]
        plt.bar(headers, values)
        plt.ylabel("processing time in seconds")
        plt.xlabel("Method")
        plt.xticks(rotation=90)
        plt.suptitle(f"[{DS}] Performance of inhouse and sklearn", fontsize=16)
        plt.title("Comparing {}".format(", ".join(df["method"].unique())), fontsize=12)
        save_figure_as(plt, get_path_to_save_plot_for(DS, TYPE_KNN, f"performance"))

#plot_knn_scores()


def plot_gd_scores():
    for DS in DATA_ALL:
        df = get_dataframe_from_file(get_path_of_set_for(DS, TYPE_GD))
        df_headers = list(df.columns.values)
        print(df_headers)
        legend = []
        n = 0
        for lR in reversed(df["learning_rate"].unique()):
            for metric in ["mse", "mae", "r_two"]:
                filt1 = df["learning_rate"] == lR
                for method_name in df["method"].unique():
                    filt2 = df["method"] == method_name
                    plt.plot('nepoch', metric, data=df.loc[filt1 & filt2], color=COLORS[n], ls=['-','--','-.',':'][n%4] , linewidth=2, alpha=0.7)
                    legend.append(method_name)
                    n+=1
                
                plt.legend(legend, fontsize="medium")
                plt.ylabel(metric)
                plt.xlabel("Number of Epochs")
                plt.suptitle(f"[{DS}] GD Evaluation of {metric} ", fontsize=16)
                plt.title("Comparing {}".format(", ".join(df["method"].unique())), fontsize=12)
                legend = []
                n = 0
                
                save_figure_as(plt, get_path_to_save_plot_for(DS, TYPE_GD, f"{lR}_{metric}"))

        df.drop(columns=["learning_rate", "mse", "mae", "r_two"], inplace=True)
        values = [v[0] for v in df.groupby(["method", "nepoch"]).mean().values]
        headers = ["inhouse\n1", "inhouse\n10", "inhouse\n100", "inhouse\n1000", "inhouse\n10000", "sklearn\n1", "sklearn\n10", "sklearn\n100", "sklearn\n1000", "sklearn\n10000"]
        plt.bar(headers, values)
        plt.ylabel("processing time in seconds")
        plt.xlabel("Method")
        plt.xticks(rotation=90)
        plt.suptitle(f"[{DS}] Performance of inhouse and sklearn", fontsize=16)
        plt.title("Comparing {}".format(", ".join(df["method"].unique())), fontsize=12)
        save_figure_as(plt, get_path_to_save_plot_for(DS, TYPE_GD, f"performance"))

plot_gd_scores()
plot_knn_scores()