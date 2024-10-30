import pandas as pd
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

file_path = os.path.join("history-search", "data", "company_invoice_details.csv")

class Expriment:

    def __init__(self):        
        self.data = None
        self.models = "history_search"
        self.group_iter = 0

    def load_data(self, file_path:os.path=file_path):
        self.data = pd.read_csv(os.path.join(file_path), low_memory=False)
        self.group = self.data.groupby(["CompanyID", "InvoiceIssuerID", "IssuerArticleID"])
        return self
    
    def daterange(self, date1, date2):
        for n in range(int((date2 - date1) / np.timedelta64(1, 'D')) + 1):
            yield date1 + np.timedelta64(n, 'D')
    
    def history_search(self, X_train, y_train, X_test):
        postives = X_train[y_train>0]
        train_days = set(postives['day'])
        train_dayofweeks = set(postives['dayofweek'])
        
        def predict_row(row):
            return row['day'] in train_days or row['dayofweek'] in train_dayofweeks 
        predictions = X_test.apply(predict_row, axis=1)
    
        return predictions

    def map_at_1(self, y_true, y_pred):
        if np.sum(y_pred) == 0:
            return 0.0  # If no positive predictions, precision is 0
        true_positive = np.sum(y_true[y_pred])
        false_positive = np.sum(~y_true[y_pred])
        return true_positive / (true_positive + false_positive)

    def recall_at_1(self, y_true, y_pred):
        if np.sum(y_true) == 0:
            return 1.0  # If no actual positives, recall is 1 by convention
        true_positive = np.sum(y_true[y_pred])
        false_negative = np.sum(y_true[~y_pred])
        return true_positive / (true_positive + false_negative)

    def f1_at_1(self, y_true, y_pred):
        precision = self.map_at_1(y_true, y_pred)
        recall = self.recall_at_1(y_true, y_pred)
        if precision + recall == 0:
            return 0.0  # Avoid division by zero
        return 2 * (precision * recall) / (precision + recall)

    def do_models(self, group):
        res = {"true_sample":len(group)}
        self.group_iter += 1
        if len(group) <= 5:  # at least 5 samples for validation
            res["total_sample"] = 0
            res[self.models + "_map@1"] = 0
            res[self.models + "_recall@1"] = 0
            res[self.models + "_f1@1"] = 0
        else:
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

            split = int(len(product_train_df) * 0.8)
            X_train, y_train = X[:split], y[:split]
            X_test, y_test = X[split:], y[split:]

            y_pred = self.history_search(X_train, y_train, X_test)

            res[self.models + "_map@1"] = self.map_at_1(y_test, y_pred)
            res[self.models + "_recall@1"] = self.recall_at_1(y_test, y_pred)
            res[self.models + "_f1@1"] = self.f1_at_1(y_test, y_pred)

        return pd.Series(res)
    
    def run(self):
        return self.group.apply(self.do_models).reset_index()

exp = Expriment()
exp = exp.load_data()
exp_result = exp.run()
exp_result.to_csv("history-search-top1.csv", index=False)
