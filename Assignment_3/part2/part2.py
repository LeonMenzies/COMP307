import pandas as pd

from sklearn.preprocessing import LabelEncoder

# def process_data(data):

#     # Remove count row
#     d = data.drop(data.columns[[0]], axis=1)

#     # Remove labels
#     labels = d.iloc[:, -1]

#     # Seperate the data from the labels
#     instances = d.iloc[:, :-1]

#     return instances, labels


# X, y = process_data(pd.read_csv('part2/breast-cancer-training.csv'))


def naive_bayes(df):

    count = {}

    for index, row in df.iterrows():

        for index, feature in row.items():
            
            print(feature)
        

naive_bayes(pd.read_csv('part2/breast-cancer-training.csv'))


