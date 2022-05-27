To Execute the Script:

1- Please change the arguments of read_csv functions located in lines 143 to 146
   to pathways of csv files IF THEY ARE NOT LOCATED IN THE SAME DIRECTORY WITH THE SCRIPT.

   eg. xtrain = pd.read_csv('path_to_xtrain/xtrain.csv')

2- If you want to run script and read outputs without script exitting after execution, you can
   start the command line in the directory script is located and type python q3main.py. That
   way, a text editor is not needed.

To Read Outputs:
1- Results are printed for each part of the question in the following format:

---- Q3 Part 2 Multinomial Naive Bayes With using MLE as Likelihood ----
Confusion Matrix:
  [[289.  28.]
 [ 15. 703.]]
Accuracy: 0.9584541062801932
Recall: 0.9116719242902208
Precision: 0.9506578947368421
F-Score: 0.9307568438003221
Specifity: 0.979108635097493
Number of wrong predictions: 43
