
##########################################
# Salary Estimation with Machine Learning
###########################################

######################
# Business Problem
######################

# Develop a machine learning model to estimate the salaries of baseball players
# whose salary information and career statistics for 1986 are shared.

#########################
# About Dataset
#########################

# Dataset 1988 ASA Graphics Section. It is part of the data used in the Poster Session.
# Salary data originally taken from Sports Illustrated, April 20, 1987.
# 1986 and career statistics, 1987 Baseball Encyclopedia published by Collier Books,
# Macmillan Publishing Company, New York obtained from the update.

# 20 Features 322 Observations 21 KB

# AtBat: Number of hits with a baseball bat in the 1986-1987 season
# Hits: Number of hits in the 1986-1987 season
# HmRun: Most valuable hits in the 1986-1987 season
# Runs: The points (s)he earned for his team in the 1986-1987 season
# RBI: Number of players a hitter had jogged when (s)he hit
# Walks: Number of mistakes made by the opposing player
# Years: Player's playing time in major league (years)
# CAtBat: Number of hits during the player's career
# CHits: Number of hits made by the player throughout his/her career
# CHmRun: The player's most valuable point during his/her career
# CRuns: The number of points the player has earned for his/her team during his career
# CRBI: Number of players the player has run during his/her career
# CWalks: Number of mistakes made by the opposing player during the player's career
# League: A factor with A and N levels showing the league in which the player played until the end of the season
# Division: A factor with E and W levels indicating the position played by the player at the end of 1986
# PutOuts: Helping your teammate during the game
# Assits: The number of assists the player made in the 1986-1987 season
# Errors: The number of errors of the player in the 1986-1987 season
# Salary: Salary of the player in 1986-1987 season (over thousand)
# NewLeague: A factor with A and N levels showing the player's league at the start of the 1987 season

##############################################
# Salary Prediction with Multiple Linear Regression
##############################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor

##################################################################################
#                      1. Exploratory Data Analysis
##################################################################################

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("Projects/Baseball_Salary_Estimate/hitters.csv")

#------------------Overview-----------------------------------------


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


#-------------------Numerical and Categorical Variable Analysis---------------------
TARGET = "Salary"

def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Analysis of categorical variables;


def cat_variable_overview(dataframe, cat_cols):
    for col in cat_cols:
        print("-------", col,"------------")
        print(dataframe[col].value_counts())
        plt.title("Distribution of Variable")
        sns.countplot(x=col, data=df)
        plt.xlabel(col)
        plt.show(block=True)

cat_variable_overview(df, cat_cols)


# Analysis of numerical variables;


def num_variable_overview(dataframe, num_cols):
    print(dataframe[num_cols].describe().T)
    for col in num_cols:
        plt.hist(dataframe[col])
        plt.title("Distribution of Variable")
        plt.xlabel(col)
        plt.show(block=True)

num_variable_overview(df, num_cols)


# Average of numerical variables relative to the dependent variable;


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "Salary", col)


# Mean of categorical variables by dependent variable;


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)


# Correlation analysis;


def correlation(dataframe, plot=False):
    corr_matrix = dataframe.corr()

    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(figsize=[20, 15])
        sns.heatmap(dataframe.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
        ax.set_title("Correlation Matrix", fontsize=20)
        plt.show(block=True)
    return print(corr_matrix)

correlation(df, plot=True)


#####################################################################################
#                  2. Data Preprocessing & Feature Engineering
#####################################################################################

df_ = df.copy()

# Outlier threshold for variables;

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


#  Outliers check;


def check_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in df_.columns:
    print(col, check_outlier(df_, num_cols))

# No outliers in defined ranges for numeric variables.


# Missing value and ratio analysis;


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


na_cols = missing_values_table(df_, True)

#         n_miss  ratio
# Salary      59  18.32

# Since the missing data is in the variable to be estimated, it was not filled in.
df_.dropna(inplace=True)
#--------------------feature extraction----------------------------------------------

# Percentage of most valuable hits in the 1986-1987 season
df_["HMRUN_ATBAT"] = df_["HmRun"] / df_["AtBat"] * 100
# The ratio of the number of points the player earned to his team in the 1986-1987 season in his career
df_["RUN_RATE"] = df_["Runs"] / df_["CRuns"]
# The rate of hits in his career in the 1986-1987 season
df_["HITS_RATE"] = df_["Hits"] / df_["CHits"]
# Ratio of the most valuable hits in his career in the 1986-1987 season
df_["HMRUN_RATE"] = df_["HmRun"] / (df_["CHmRun"] + 0.0001)
# The ratio of the number of hits to the career in the 1986-1987 season
df_["ATBAT_RATE"] = df_["AtBat"] / df_["CAtBat"]
# Ratio of the number of players a batsman runs when he strikes, in his career
df_["RBI_RATE"] = df_["RBI"] / df_["CRBI"]
# The ratio of the number of mistakes made by the opposing player in his career
df_["Walks_RATE"] = df_["Walks"] / df_["CWalks"]

check_df(df_)

#-----------------------Processing Encoding------------------------------------------

# Standardization
num_cols = [col for col in num_cols if 'Salary' not in col]

ss = StandardScaler()

df_[num_cols] = ss.fit_transform(df_[num_cols])

df_.head()


# Label Encoding;

le = LabelEncoder()

binary_cols = [col for col in df_.columns if df_[col].dtype not in [int, float]
               and df_[col].nunique() == 2]

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


for col in binary_cols:
    df_ = label_encoder(df_, col)

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df_.columns if 10 >= df_[col].nunique() > 2]

df_ = one_hot_encoder(df_, ohe_cols)


df_.head()



################################################################################
#                   3. Modelling
################################################################################

# Linear Regression Model
y = df_["Salary"]
X = df_.drop("Salary", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
linreg = LinearRegression()
model = linreg.fit(X_train,y_train)


y_pred = model.predict(X_train)
lin_train_rmse =np.sqrt(mean_squared_error(y_train,y_pred))
print("LINEAR REGRESSION TRAIN RMSE:", "{:,.2f}".format(np.sqrt(mean_squared_error(y_train,y_pred))))


lin_train_r2 = linreg.score(X_train,y_train)
print("LINEAR REGRESSION TRAIN R-SQUARED:", "{:,.3f}".format(linreg.score(X_train,y_train)))


y_pred = model.predict(X_test)
lin_test_rmse =np.sqrt(mean_squared_error(y_test,y_pred))
print("LINEAR REGRESSION TEST RMSE:", "{:,.2f}".format(np.sqrt(mean_squared_error(y_test,y_pred))))


lin_test_r2 = linreg.score(X_test,y_test)
print("LINEAR REGRESSION TEST R-SQUARED:", "{:,.3f}".format(linreg.score(X_test,y_test)))


# Test part regplot:

g = sns.regplot(x=y_test, y=y_pred, scatter_kws={'color': 'b', 's': 5},
                ci=False, color="r")
g.set_title(f"Test Model R2: = {linreg.score(X_test, y_test):.3f}")
g.set_ylabel("Predicted Salary")
g.set_xlabel("Salary")
plt.xlim(-5, 2700)
plt.ylim(bottom=0)
plt.show(block=True)

print("LINEAR REGRESSION CROSS_VAL_SCORE:", "{:,.3f}".format(np.mean(np.sqrt(-cross_val_score(model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))))


# Random Forest Model

y = df_["Salary"]
X = df_.drop("Salary", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=1)


rf_model = RandomForestRegressor().fit(X_train, y_train)

y_pred1 = rf_model.predict(X_train)

y_pred2 = rf_model.predict(X_test)

rf_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred1))
rf_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred2))
rf_train_r2 = r2_score(y_train, y_pred1)
rf_test_r2 = r2_score(y_test, y_pred2)

print("RF Train RMSE:", "{:,.2f}".format(np.sqrt(mean_squared_error(y_train, y_pred1))))
print("RF Test RMSE:", "{:,.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred2))),"\n")
print("RF Train MAE:", "{:,.2f}".format(mean_absolute_error(y_train, y_pred1)))
print("RF Test MAE:", "{:,.2f}".format(mean_absolute_error(y_test, y_pred2)), "\n")
print("RF Train R^2:", "{:,.2f}".format(r2_score(y_train, y_pred1)))
print("RF Test R^2:", "{:,.2f}".format(r2_score(y_test, y_pred2)))


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train)

# ----------Linear Regression Model
# LINEAR REGRESSION TRAIN RMSE: 209.38
# LINEAR REGRESSION TRAIN R-SQUARED: 0.703

# LINEAR REGRESSION TEST RMSE: 300.81
# LINEAR REGRESSION TEST R-SQUARED: 0.625

# LINEAR REGRESSION CROSS_VAL_SCORE: 255.628

# ---------Random Forest Model
# RF Train RMSE: 90.90
# RF Test RMSE: 303.08

# RF Train MAE: 48.91
# RF Test MAE: 160.88

# RF Train R^2: 0.94
# RF Test R^2: 0.62