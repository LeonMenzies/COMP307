import pandas as pd
import numpy as np
import sys

#Save data to dataframes
df_train = pd.read_csv("./part2/hepatitis-training", delimiter=r"\s+");
df_test = pd.read_csv("./part2/hepatitis-test", delimiter=r"\s+");

class Node:
    def __init__(self, attribute, depth, probability = None, left_node = None, right_node = None, is_leaf = False, node_class = None):
        self.attribute = attribute
        self.left_node = left_node
        self.right_node = right_node 
        self.is_leaf = is_leaf
        self.depth = depth   
        self.probability = probability
        self.node_class = node_class

def get_prob(data):
    return data.Class.value_counts().die + data.Class.value_counts().live / len(data)


def build_tree(instances, attributes, depth):

    #Increase depth for printing
    depth += 1

    if len(instances) == 0:
        return Node("Leaf", depth, is_leaf=True, node_class="live", probability=get_prob(df_train))
    elif calc_impurity(instances, len(instances)) == 0:
        return Node("Leaf", depth, is_leaf=True, node_class="live")
    elif len(attributes) == 1:
        return Node("Leaf", depth, is_leaf=True, node_class="live")
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

        left = build_tree(best_true, attributes, depth)
        right = build_tree(best_false, attributes, depth)

        return Node(best_att, depth, left, right)
    

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

    if(node.is_leaf):
        print(f"Class  prop = {node.probability}")
        return

    left = node.left_node
    right = node.right_node

    if left != None:
         print_tree(left)
    if right != None:
         print_tree(right)


   
print_tree(root_node)