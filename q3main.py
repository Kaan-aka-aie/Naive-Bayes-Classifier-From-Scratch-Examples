import pandas as pd
import numpy as np
np.seterr(all="ignore")

def Q3_Part2(xtrain: pd.DataFrame, ytrain: pd.DataFrame , xtest: pd.DataFrame, ytest: pd.DataFrame) -> np.array:
    """
    Training and Testing multinomial naive Bayes Classifier by using MLE for likelihood.
    Returns Confusion matrix.
    """
    # Separating data to classes
    xspam = xtrain[ytrain["Prediction"] == 1]
    xnormal = xtrain[ytrain["Prediction"] == 0]
    # --- TRAINING --- #
    # Determining priors (P(Y = y_k)) (pi values)
    # And Taking their logarithms
    pSpam = np.log(xspam.shape[0]/xtrain.shape[0])
    pNormal = np.log(xnormal.shape[0]/xtrain.shape[0])

    # Determining Likelihoods (P(w_t|spam) & P(w_t|normal) as separate arrays/vectors) (thetas)
    # And taking their logarithms
    theta_spam = np.log(xspam.sum().to_numpy()/xspam.sum().sum())
    theta_normal = np.log(xnormal.sum().to_numpy()/xnormal.sum().sum())
    # Making - inf values -10^12
    theta_spam[np.isneginf(theta_spam)] = -(10**12)
    theta_normal[np.isneginf(theta_normal)] = -(10 ** 12)

    # --- TESTING --- #
    X = xtest.to_numpy() # getting the vector x for each document
    pSpam_X = np.sum(X * theta_spam, axis=1) + pSpam # getting ln(prior) + sigma[t_i_j * ln(likelihood)] for spam
    pNormal_X = np.sum(X * theta_normal,axis=1)  + pNormal # getting ln(prior) + sigma[t_i_j * ln(likelihood)] for normal

    # Comparing predictions with y_test
    predictions = []
    for j in range(pSpam_X.shape[0]):
        if pSpam_X[j] > pNormal_X[j]:
            predictions.append(1)
        else:
            predictions.append(0)

    ytest["bayes_pred"] = predictions
    labels = [1,0]
    conf = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            conf[i, j] = np.sum((ytest["bayes_pred"] == labels[i]) & (ytest["Prediction"] == labels[j]))

    return conf


def Q3_Part3(xtrain: pd.DataFrame, ytrain: pd.DataFrame , xtest: pd.DataFrame, ytest: pd.DataFrame, alpha:int) -> np.array:
    """
       Training and Testing multinomial naive Bayes Classifier by using MAP for likelihood.
       Returns Confusion matrix.
    """
    # Separating data to classes
    xspam = xtrain[ytrain["Prediction"] == 1]
    xnormal = xtrain[ytrain["Prediction"] == 0]

    # --- TRAINING --- #
    # Determining priors (P(Y = y_k)) (pi values)
    # And Taking their logarithms
    pSpam = np.log(xspam.shape[0] / xtrain.shape[0])
    pNormal = np.log(xnormal.shape[0] / xtrain.shape[0])

    # Determining Likelihoods (P(w_t|spam) & P(w_t|normal) as separate arrays/vectors) (thetas)
    # And taking their logarithms
    theta_spam = np.log( (xspam.sum().to_numpy() + alpha) / (xspam.sum().sum() + alpha * xspam.shape[1]))
    theta_normal = np.log((xnormal.sum().to_numpy()+alpha) / (xnormal.sum().sum()+ alpha * xspam.shape[1]))
    # Making - inf values -10^12
    theta_spam[np.isneginf(theta_spam)] = -(10 ** 12)
    theta_normal[np.isneginf(theta_normal)] = -(10 ** 12)

    # --- TESTING --- #
    X = xtest.to_numpy()  # getting the vector x for each document
    pSpam_X = np.sum(X * theta_spam, axis=1) + pSpam  # getting ln(prior) + sigma[t_i_j * ln(likelihood)] for spam
    pNormal_X = np.sum(X * theta_normal,axis=1) + pNormal # getting ln(prior) + sigma[t_i_j * ln(likelihood)] for normal

    # Comparing predictions with y_test
    predictions = []
    for j in range(pSpam_X.shape[0]):
        if pSpam_X[j] > pNormal_X[j]:
            predictions.append(1)
        else:
            predictions.append(0)

    ytest["bayes_pred"] = predictions
    labels = [1, 0]
    conf = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            conf[i, j] = np.sum((ytest["bayes_pred"] == labels[i]) & (ytest["Prediction"] == labels[j]))

    return conf

def Q3_Part4(xtrain: pd.DataFrame, ytrain: pd.DataFrame , xtest: pd.DataFrame, ytest: pd.DataFrame) -> np.array:
    """
           Training and Testing binomial naive Bayes Classifier by using MLE for likelihood.
           Returns Confusion matrix.
    """
    # Converting data to binary data
    xtrain[:] = np.where(xtrain >= 1,1,0)
    xtest[:] = np.where(xtest >= 1,1,0)
    # Separating data to classes
    xspam = xtrain[ytrain["Prediction"] == 1]
    xnormal = xtrain[ytrain["Prediction"] == 0]

    # --- TRAINING --- #
    # Determining priors (P(Y = y_k)) (pi values)
    # And Taking their logarithms
    pSpam = np.log(xspam.shape[0] / xtrain.shape[0])
    pNormal = np.log(xnormal.shape[0] / xtrain.shape[0])
    # Determining Likelihoods (P(w_t|spam) & P(w_t|normal) as separate arrays/vectors) (thetas)
    # This time not taking their logarithm. (It has to be taken with S_ij multiplier for computation)
    theta_spam = xspam.sum().to_numpy() / xspam.shape[0]
    theta_normal = xnormal.sum().to_numpy() / xnormal.shape[0]
    # Making - inf values -10^12
    theta_spam[np.isneginf(theta_spam)] = -(10 ** 12)
    theta_normal[np.isneginf(theta_normal)] = -(10 ** 12)

    # --- TESTING --- #
    X = xtest.to_numpy()  # getting the vector x for each document
    pSpam_X = np.sum(np.log(X * theta_spam + (1-X) * (1-theta_spam)) , axis=1) + pSpam  # getting ln(prior) + sigma[t_i_j * ln(likelihood)] for spam
    pNormal_X = np.sum( np.log(X * theta_normal + (1-X) * (1- theta_normal)),axis=1) + pNormal  # getting ln(prior) + sigma[t_i_j * ln(likelihood)] for normal

    # Comparing predictions with y_test
    predictions = []
    for j in range(pSpam_X.shape[0]):
        if pSpam_X[j] > pNormal_X[j]:
            predictions.append(1)
        else:
            predictions.append(0)

    ytest["bayes_pred"] = predictions
    labels = [1, 0]
    conf = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            conf[i, j] = np.sum((ytest["bayes_pred"] == labels[i]) & (ytest["Prediction"] == labels[j]))

    return conf

xtrain = pd.read_csv("x_train.csv")
ytrain = pd.read_csv("y_train.csv")
xtest= pd.read_csv("x_test.csv")
ytest = pd.read_csv("y_test.csv")

print("---- Q3 Part 2 Multinomial Naive Bayes With using MLE as Likelihood ----")
confusion1 = Q3_Part2(xtrain,ytrain,xtest,ytest)
ac1 = (confusion1[0,0] + confusion1[1,1])/confusion1.sum()
recall1 = confusion1[0,0]/(confusion1[0,0] + confusion1[0,1])
precission1 = confusion1[0,0]/(confusion1[0,0] + confusion1[1,0])
fscore1 = 2 * recall1 * precission1/(recall1 + precission1)
spec1 = confusion1[1,1]/(confusion1[1,0] + confusion1[1,1])
wrong = confusion1[0,1] + confusion1[1,0]
print("Confusion Matrix:\n ",confusion1)
print("Accuracy:",ac1)
print("Recall:", recall1)
print("Precision:", precission1)
print("F-Score:",fscore1)
print("Specifity:",spec1 )
print("Number of wrong predictions:", wrong)

print("---- Q3 Part 3 Multinomial Naive Bayes With using MAP as Likelihood ----")
confusion1 = Q3_Part3(xtrain,ytrain,xtest,ytest,5)
ac1 = (confusion1[0,0] + confusion1[1,1])/confusion1.sum()
recall1 = confusion1[0,0]/(confusion1[0,0] + confusion1[0,1])
precission1 = confusion1[0,0]/(confusion1[0,0] + confusion1[1,0])
fscore1 = 2 * recall1 * precission1/(recall1 + precission1)
spec1 = confusion1[1,1]/(confusion1[1,0] + confusion1[1,1])
wrong = confusion1[0,1] + confusion1[1,0]
print("Confusion Matrix:\n ",confusion1)
print("Accuracy:",ac1)
print("Recall:", recall1)
print("Precision:", precission1)
print("F-Score:",fscore1)
print("Specifity:",spec1 )
print("Number of wrong predictions:", wrong)

print("---- Q3 Part 4 Binomial Naive Bayes With using MLE as Likelihood ----")
confusion1 = Q3_Part4(xtrain,ytrain,xtest,ytest)
ac1 = (confusion1[0,0] + confusion1[1,1])/confusion1.sum()
recall1 = confusion1[0,0]/(confusion1[0,0] + confusion1[0,1])
precission1 = confusion1[0,0]/(confusion1[0,0] + confusion1[1,0])
fscore1 = 2 * recall1 * precission1/(recall1 + precission1)
spec1 = confusion1[1,1]/(confusion1[1,0] + confusion1[1,1])
wrong = confusion1[0,1] + confusion1[1,0]
print("Confusion Matrix:\n ",confusion1)
print("Accuracy:",ac1)
print("Recall:", recall1)
print("Precision:", precission1)
print("F-Score:",fscore1)
print("Specifity:",spec1 )
print("Number of wrong predictions:", wrong)
