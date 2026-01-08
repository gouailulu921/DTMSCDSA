from param import parameter_parser
from DTMSCDSA import DTMSCDSA
from dataprocessing import data_pro
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import random
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import pandas as pd

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')

def lossA(score,LDA):
    B = np.random.uniform(1,2,(LDA.shape))
    B = torch.from_numpy(B)
    C = np.argwhere(LDA == 0)
    for i in range(len(C)):
        B[C[i][0],C[i][1]] = 1
    L = (score-LDA)*B
    loss = torch.norm(L)
    return loss

def train(model, train_data, optimizer, opt):
    model.train()
    LOSS = []
    iteration = []
    for epoch in range(0, opt.epoch):
        model.zero_grad()
        cicrRNA_feature, drug_feature, score = model(train_data)
        loss = lossA(score, train_data['cd_p'])
        LOSS.append(loss.item())
        iteration.append(epoch)
        loss.backward()
        optimizer.step()
        print(loss.item())
    # plt.plot(LOSS)
    # plt.title('Loss over epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.grid()
    # plt.show()
    return cicrRNA_feature, drug_feature, score

def main():
    args = parameter_parser()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = data_pro(args)
    train_data = dataset
    model = DTMSCDSA(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    cicrRNA_feature, drug_feature, score = train(model, train_data, optimizer, args)
    print(cicrRNA_feature)
    print(dataset['one_index'])
    posi_list = []
    for i in range(len(dataset['one_index'])):
        posi = cicrRNA_feature[dataset['one_index'][i][0], :].tolist() + drug_feature[dataset['one_index'][i][1], :].tolist() + [1, 0]
        posi_list.append(posi)
    unlabelled_list = []
    for i in range(len(dataset['zero_index'])):
        unlabelled = cicrRNA_feature[dataset['zero_index'][i][0], :].tolist() + drug_feature[dataset['zero_index'][i][1], :].tolist() + [0, 1]
        unlabelled_list.append(unlabelled)
    unlabelled_list = random.sample(unlabelled_list, len(posi_list))
    posi_data = np.array(posi_list)
    nega_data = np.array(unlabelled_list)
    print(posi_data.shape)
    print(nega_data.shape)
    data = np.concatenate((posi_data, nega_data))
    X = data[:, :-2]
    y = data[:, -2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    xgb = XGBClassifier()
    xgb.load_model(".\model_1.json")
    xgb.fit(X_train, y_train)
    pred_final = xgb.predict(X_test)
    y_score_x = xgb.predict_proba(X_test)[:, 1]
    fpr_x, tpr_x, thresholds = roc_curve(y_test, y_score_x)
    roc_auc_x = auc(fpr_x, tpr_x)
    print('auc', roc_auc_x)
    p_x, r_x, _ = precision_recall_curve(y_test, y_score_x)
    AUPR_x = auc(r_x, p_x)
    print('AUPR', AUPR_x)
    mcc = metrics.matthews_corrcoef(y_test, pred_final)
    print('mcc', mcc)
    f1_score = metrics.f1_score(y_test, pred_final)
    print('f1-score', f1_score)
    from sklearn.metrics import precision_score, recall_score
    precision = precision_score(y_test, pred_final)
    recall = recall_score(y_test, pred_final)
    print("Precision:", precision)
    print("Recall:", recall)
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_test, pred_final).ravel()

    sensitivity = tp / (tp + fn)
    specifcity = tn / (tn + fp)

    print("Sensitivity: ", sensitivity)
    print("Specifcity: ", specifcity)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, pred_final)
    print("ACC =", acc)
if __name__ == "__main__":
    main()

