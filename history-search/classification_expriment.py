import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error
import os
import numpy as np
import xgboost
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

file_path = os.path.join("history-search", "data", "company_invoice_details.csv")

class Expriment:

    def __init__(self):        
        self.data = None
        self.models = [ "history_search", "xgboost", "RandomForestClassifier", "naive_bayes", "linear"]
        self.group_iter = 0

    def load_data(self, file_path:os.path=file_path):
        self.data = pd.read_csv(os.path.join(file_path), low_memory=False)
        self.group = self.data.groupby(["CompanyID", "InvoiceIssuerID", "IssuerArticleID"])
        return self
    
    def daterange(self, date1, date2):
        for n in range(int((date2 - date1) / np.timedelta64(1, 'D')) + 1):
            yield date1 + np.timedelta64(n, 'D')

    def history_search(self, X_test,  X_train, y_train):
        postives = X_train[y_train>0]
        train_days = set(postives['day'])
        train_dayofweeks = set(postives['dayofweek'])
        
        def predict_row(row):
            return row['day'] in train_days or row['dayofweek'] in train_dayofweeks 
        predictions = X_test.apply(predict_row, axis=1)
    
        return predictions
    
    def do_models(self, group):
        res = {"true_sample": len(group)}
        self.group_iter += 1
        if len(group) <= 5:  # at least 5 samples for validation
            res["total_sample"] = 0
            for model in self.models:
                res[model] = -1
        else:
            # Create X, y for Linear model, then for other models
            X_linear = group[["dayofweek", "day", "month"]]
            y_linear = group["Total"]
            X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(X_linear, y_linear, test_size=0.2, shuffle=False)

            durtaion = np.unique(pd.to_datetime(group["InvoiceDate"]))
            inovice_start = durtaion[0]
            invoice_end = durtaion[-1]
            new_dates = []
            for date in self.daterange(inovice_start, invoice_end):
                panda_date = pd.DatetimeIndex([date])
                new_dates.append([
                    date, panda_date.dayofweek[0], panda_date.day[0],
                    panda_date.month[0],
                    (durtaion == date).any()
                ])
            product_train_df = pd.DataFrame(new_dates, columns=["InvoiceDate", "dayofweek", "day",
                                                    "month", "bought"])
            X = product_train_df[["dayofweek", "day", "month"]]
            y = product_train_df["bought"].to_numpy()
            res["total_sample"] = len(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            for model in self.models:
                if model == "linear":
                    model_object = LinearRegression()
                    model_object.fit(X_train_linear, y_train_linear)
                    y_pred = model_object.predict(X_test_linear)
                    result = mean_squared_error(y_test_linear, y_pred)
                    res[model] = result

                elif model =="history_search":
                    y_pred = self.history_search(X_test, X_train, y_train)
                else:
                    if model == "xgboost":
                        model_object = xgboost.XGBClassifier()
                    elif model =="RandomForestClassifier":
                        model_object = RandomForestClassifier()
                    elif model =="naive_bayes":
                        model_object = ComplementNB()
                    else:
                        print (f"Model ({model}) is not implemented")
                    model_object.fit(X_train, y_train)
                    y_pred = model_object.predict(X_test)
                        
                if model != "linear":
                    result = f1_score(y_test, y_pred).item()
                    res[model] = result

        return pd.Series(res)
    
    def run(self):
        return self.group.apply(self.do_models)

exp = Expriment()
exp = exp.load_data()
exp_result = exp.run()
exp_result.to_csv("result.csv")
