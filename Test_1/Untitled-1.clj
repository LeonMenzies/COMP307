COMP307/AIML420 Test 1
Answer Sheet

Name: Leon Menzies  
Student ID: 300543278

------------------------------------------------------------------------------------------

Question 1. Search [5 marks]

(a) [3 marks]
  1. F
  2. F
  3. T

(b) [2 marks]
You can get stuck in a local optima, simulated annealing occasionaly takes incorrect steps to try and
avoid this.


------------------------------------------------------------------------------------------

Question 2. Machine Learning [15 marks]

(a) [3 marks]
Regression is the output of variables where as classification is the output of a class label.
Regression: calculating the potential cost of houses in 5 years
classification: If the given animal is a dog or a cat


(b) [3 marks]
Choose a number of k-folds, e.g 5 fold. iterate 5 times and create 5 different splits of training and test data to train and test the 
the accuracy of the model. take the average of all 5 runs to get the accuracy


(c)
  (i) [2 marks]

    1/6 for yes * 3/6 for no
    Avg impurity: 0.08333
  
  (ii) [2 marks]
    It would be a better feature to choose as it has a lower impurity 

(d) [5 marks]
Overfitting is when the model has learned too much from the data so it is too specific to the training data.
underfitting is when it doesnt learn enough to create good predictions

A way to prevent this is using Kfold or testing on a subset of the training data to find where the accuracy is best before it drops off again
and becomes overfitted. You can stop at this point with confidence it has not over or under fitted.

------------------------------------------------------------------------------------------

Question 3. Neural Networks [8 marks]

(a)
3 = 1.5 + -2 + 0.5 = 0
4 = -0.5 + 1 + -0.5 = 1

5 = 0 + -1 + 0 = -1
6 = 0 + -1 + 1 = 0

o5 = -1

(b)
Betas = -1 + 0.5
Deltas = 0.2 * 1 * (1 - 1) * Betas




------------------------------------------------------------------------------------------

Question 4. Evolutionary Computation [12 marks]

(a) [6 marks]
Genetic algorithms is a search where the best indivduals are selected from each evolution much like natural selection.
These best indivduals are used in the next steop until a required threshold is met

(b) [6 marks]
(i) Terminal set = given feature values
(ii) Functional set = Addition and subtraction
(iii) If the output is greater then a certain amount its X otherwise O
(iv) Same as iii but with 3 output options




