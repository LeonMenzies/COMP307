import pandas as pd
data = pd.read_csv('part2/breast-cancer-training.csv')

# Remove count row
d = data.drop(data.columns[[0]], axis=1)


def naive_bayes(df):

    count = {}

    #Initialise the count numbers to 1
    #For each class label y
    for y in d['class'].unique():

        count[y] = 1

        #For each feature X
        for X in df.drop(df.columns[[0]], axis=1).columns:  

            #For each possible value x of feature X
            for x in df[X].unique():

                count[(X, x, y)] = 1


    #Count the numbers of each class and feature value based on the training instances
    for index, row in df.iterrows():
        count[row['class']] += 1

        for X in df.drop(df.columns[[0]], axis=1).columns:

            count[(X, row[X], row['class'])] += 1
        

    #Calculate the total/denominators

    class_total = 0
    total = {}
    for y in d['class'].unique():
        class_total += count[y]

        for X in df.drop(df.columns[[0]], axis=1).columns:
            
            total[(X, y)] = 0

            for x in df[X].unique():
                total[(X, y)] += count[(X, x, y)]

    prob = {}
    #Calculate probabilities from the counting numbers
    for y in d['class'].unique():
        prob[y] = count[y] / class_total

        for X in df.drop(df.columns[[0]], axis=1).columns:
            for x in df[X].unique():
                prob[(X, x, y)] = count[X, x, y] / total[(X, y)]

    return prob

#Load test dat
test_data = pd.read_csv('part2/breast-cancer-test.csv')

# Remove count row
td = test_data.drop(test_data.columns[[0]], axis=1)




#Score an instance
def score(instance, y):

    prob = naive_bayes(d)
    score = prob[y]

    for X, x in instance.items():
     score *= prob[(X, x, y)]

    return score


#Iterate the test set and get final score
total = 0
for index, row in td.iterrows():

    y = row['class']

    correct = None
    if score(row.drop(labels=['class']), 'no-recurrence-events') > score(row.drop(labels=['class']), 'recurrence-events'):
        correct = 'no-recurrence-events'
    else:
        correct = 'recurrence-events'
    if correct == y:
        total += 1


print(f"Accurracy: {total / len(td)}")