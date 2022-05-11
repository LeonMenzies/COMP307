import pandas as pd


def naive_bayes(df):

    count = {}

    for index, row in df.iterrows():

        count[row['class']] = 1

        for index, feature in row.items():

            for x in df[feature].unique():

                count[(feature, x, row['class'])] = 1


    print(count)
        
df = pd.read_csv('part2/breast-cancer-training.csv')

print(df)

print(df['tumor-size'].unique())

naive_bayes(df)


