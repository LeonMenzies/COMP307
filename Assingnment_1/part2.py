import pandas as pd
import numpy as np
import sys

#Save data to dataframes
df_train = pd.read_csv("./part2/hepatitis-training", delimiter=r"\s+");
df_test = pd.read_csv("./part2/hepatitis-test", delimiter=r"\s+");

y = df_test.drop('Class', axis=1)
y_class = df_test.Class

# y = df_train.drop('Class', axis=1)
# y_class = df_train.Class

class Node:
    def __init__(self, attribute, depth, probability = None, true_node = None, false_node = None, class_node=None):
        self.attribute = attribute
        self.true_node = true_node
        self.false_node = false_node 
        self.depth = depth   
        self.probability = probability
        self.class_node = class_node

def get_prob(data):

    try:
        die = data.Class.value_counts().die
    except:
        die = 0

    try:
        live = data.Class.value_counts().live
    except:
        live = 0

    return  "die" if die > live else "live" , (die + live) / len(data)    


def build_tree(instances, attributes, depth):

    #Increase depth for printing
    depth += 1

    if len(instances) == 0:

        #Create a leaf node using the most probable class
        class_val, prob = get_prob(df_train)
        return Node("Class", depth, probability=prob, class_node=class_val)
        
    elif calc_impurity(instances, len(instances)) == 0:

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


        for attribute in attributes:

                if attribute == "Class":
                    continue

                true_instance = instances.loc[instances[attribute] == True]
                false_instance = instances.loc[instances[attribute] == False]

                true_impurity = calc_impurity(true_instance, len(instances))
                false_impurity = calc_impurity(false_instance, len(instances))
                
                true_weighted = len(true_instance) / len(instances) * true_impurity
                false_weighted = len(false_instance) / len(instances) * false_impurity

                if(true_weighted + false_weighted < best_impurity):
                    best_impurity = true_weighted + false_weighted

                    best_att = attribute
                    best_true = true_instance
                    best_false = false_instance

        attributes.remove(best_att)

        true = build_tree(best_true, attributes, depth)
        false = build_tree(best_false, attributes, depth)

        return Node(best_att, depth, true_node=true, false_node=false)
    


    

def calc_impurity(instances, total):

    try:
        die = instances.Class.value_counts().die
    except:
        return 0

    try:
        live = instances.Class.value_counts().live
    except:
        return 0

    return (die / total) * (live / total)



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

results = []

def test_tree(node, test):
    
    if node.class_node != None:
        results.append(node.probability)
        return
    
    if test[1][node.attribute] == True:
        test_tree(node.true_node, test)
    
    if test[1][node.attribute] == False:
        test_tree(node.false_node, test)
    

#test_tree(root_node, y[:1])



for row in y.iterrows():
    test_tree(root_node, row)

print(results)

correct = 0

for i in range (len(results)):
    if results[i] == y_class[i]:
        correct += 1

print(len(results) / len(y_class))
print(correct / len(results))