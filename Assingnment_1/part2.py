import pandas as pd
import sys
import statistics

#Save data to dataframes
df_train = pd.read_csv(sys.argv[1], delimiter=r"\s+");
df_test = pd.read_csv(sys.argv[2], delimiter=r"\s+");

# y = df_test.drop('Class', axis=1)
# y_class = df_test.Class

y = df_train.drop('Class', axis=1)
y_class = df_train.Class

#Node class for tree nodes and leafs
class Node:
    def __init__(self, attribute, depth, probability = None, true_node = None, false_node = None, class_node=None):
        self.attribute = attribute
        self.true_node = true_node
        self.false_node = false_node 
        self.depth = depth   
        self.probability = probability
        self.class_node = class_node

#Function for getting the probability of the classes
def get_prob(data):

    try:
        die = data.Class.value_counts().die
    except:
        die = 0

    try:
        live = data.Class.value_counts().live
    except:
        live = 0

    best = die if die > live else live

    return  "die" if die > live else "live" , best / len(data)      


#Main funciton incharge of building the tree using the node class
def build_tree(instances, attributes, depth):

    #Increase depth for printing
    depth += 1

    if len(instances) == 0:

        #Create a leaf node using the most probable class
        class_val, prob = get_prob(df_train)
        return Node("Class", depth, probability=prob, class_node=class_val)
        
    elif calc_impurity(instances) == 0:

        #Return a pure node
        class_val, prob = get_prob(instances)
        return Node("Class", depth, probability=prob, class_node=class_val)

    elif len(attributes) == 1:
        #Return a leaf with majority class
        class_val, prob = get_prob(instances)
        return Node("Class", depth, probability=prob, class_node=class_val)
    else:

        best_impurity = sys.maxsize
        best_att = None
        best_true = None
        best_false = None

        #Iterate and find the next best attribute
        for attribute in attributes:

                if attribute == "Class":
                    continue

                true_instance = instances.loc[instances[attribute] == True]
                false_instance = instances.loc[instances[attribute] == False]

                true_impurity = calc_impurity(true_instance)
                false_impurity = calc_impurity(false_instance)

                #Calulate weighted impurity
                true_weighted = len(true_instance) / len(instances) * true_impurity
                false_weighted = len(false_instance) / len(instances) * false_impurity


                if(true_weighted + false_weighted < best_impurity):
                    best_impurity = true_weighted + false_weighted

                    best_att = attribute
                    best_true = true_instance
                    best_false = false_instance

        attributes.remove(best_att)

        true = build_tree(best_true, [x for x in attributes], depth)
        false = build_tree(best_false, [x for x in attributes], depth)

        return Node(best_att, depth, true_node=true, false_node=false)


#Calculate the impurity 
def calc_impurity(instances):

    try:
        die = instances.Class.value_counts().die
    except:
        return 0

    try:
        live = instances.Class.value_counts().live
    except:
        return 0

    return (die * live) / ((die + live) **2)

#Build the tree and save the root node
root_node = build_tree(df_train, list(df_train.columns.values), 0)

#Print Tree
def print_tree(node):

    if(node.class_node != None):
        print("   " * node.depth + f"Class {node.class_node} prob = {node.probability}")
        return

    #print true
    print("   " * node.depth + f"{node.attribute} = True")
    print_tree(node.true_node)

    #print false
    print("   " * node.depth + f"{node.attribute} = False")
    print_tree(node.false_node)


print_tree(root_node)

#Function for testing the tree
results = []
def test_tree(node, test):
    
    if node.class_node != None:
        return node.class_node
        
    
    if test[1][node.attribute] == True:
        return test_tree(node.true_node, test)
    
    if test[1][node.attribute] == False:
        return test_tree(node.false_node, test)

#Run kfold
def run_kfold():
    accuracies = []

    #Run through each fold
    for i in range(10):
        df_train = pd.read_csv(f"{sys.argv[1]}-run-{i}", delimiter=r"\s+");
        df_test = pd.read_csv(f"{sys.argv[2]}-run-{i}", delimiter=r"\s+");

        y = df_test.drop('Class', axis=1)
        y_class = df_test.Class
        
        #Build the tree
        root_node = build_tree(df_train, list(df_train.columns.values), 0)

        #test the tree
        results = []
        for row in y.iterrows():
            results.append(test_tree(root_node, row))
            
        correct = 0

        for j in range (len(results)):
            if results[j] == y_class[i]:
                correct += 1


        acc = correct / len(results)
        accuracies.append(acc)

        print(f"Kfold= {i} Accuracy: {acc}")

    print(f"Average accuracy accros 10 folds: {statistics.mean(accuracies)}")

run_kfold()


def run_test(y, y_class, type):
    results = []
    for row in y.iterrows():
        results.append(test_tree(root_node, row))

    correct = 0

    for i in range (len(results)):
        if results[i] == y_class[i]:
            correct += 1


    print(f"{type} accuracy: {correct / len(results)} | {correct}/{len(results)}")

run_test(df_train.drop('Class', axis=1), df_train.Class, "Training")
run_test(df_test.drop('Class', axis=1),  df_test.Class, "Test")