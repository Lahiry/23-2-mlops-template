import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

model = pickle.load(open("./models/model.pkl", "rb"))
one_hot_enc = pickle.load(open("./models/one_hot_enc.pkl", "rb"))

df = pd.read_csv("./data/bank_predict.csv")

X_test = pd.DataFrame(one_hot_enc.transform(df), columns=one_hot_enc.get_feature_names_out())
y_test = pd.read_csv("./data/bank_preprocessed.csv")["deposit"]

y_pred = model.predict(X_test)

df['y_pred'] = y_pred

# Map the predictions to "yes" or "no"
df['y_pred'] = df['y_pred'].map({1: "yes", 0: "no"})

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn+fp)

print(f"Specificity: {specificity}")

print(classification_report(y_test, y_pred))

# Save the updated DataFrame back to the CSV file
df.to_csv("./data/bank_predict.csv", index=False)