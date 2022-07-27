import pandas as pd
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
import xgboost
import lightgbm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from Extract_feature import *
import time


def Load_data():
    data = np.array(pd.read_excel('Dataset_Variants.xlsx', sheet_name='Dataset_Variants'))
    Seq = np.array(pd.read_excel('Dataset_Variants.xlsx', sheet_name='Sequence'))
    Sequence = [i[-1] for i in Seq]
    # print(len(Sequence), Sequence[-1])
    # physichemical property
    pp = np.load('feature_data.npy')
    Input = []
    Features_pp = []
    Label = []
    for i in range(len(data)):
        if data[i][0] == 'gb1':
            dex = int(data[i][3])
            Mystr = Sequence[5][:dex-1] + data[i][2] + Sequence[5][dex:]
            Input.append(Mystr[227:282])
            Features_pp.append(pp[i])
            Label.append(float(data[i][-1]))
    # print(len(Input), len(Label))
    # with open('Input.fasta', 'w') as myfile:
    #     for i in range(len(Input)):
    #         myfile.write('>'+str(i+1)+'\n'+Input[i]+'\n')
    return Features_pp, Input, Label


def ALL_features(Sequence):
    tic = time.time()
    Handcrafted, _ = Get_features(Sequence)
    print('Hand_crafted features time:', time.time()-tic)
    sequences_Example = []
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i])-1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        sequences_Example.append(zj)
    # Automatic extracted features
    tokenizer = T5Tokenizer.from_pretrained("prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("prot_t5_xl_uniref50")
    gc.collect()
    print(torch.cuda.is_available())
    # 'cuda:0' if torch.cuda.is_available() else
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    features = []
    tic = time.time()
    for i in range(len(sequences_Example)):
        # print('For sequence ', str(i+1))
        sequences_Example_i = sequences_Example[i]
        sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
        ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]
            features.append(seq_emd)
    features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)
    for i in range(len(features)):
        for k in range(len(features[0][0])):
            for j in range(len(features[i])):
                features_normalize[i][k] += features[i][j][k]
            features_normalize[i][k] /= len(features[i])
    # print(len(features_normalize), len(features_normalize[0]))
    print('Embedding features time:', time.time()-tic)
    # features_output = np.concatenate((features_normalize, Handcrafted), axis=1)
    return features_normalize, Handcrafted


def Validations(features, Label):
    for Ifeature in features:
        Ifeature = np.array(Ifeature)
        Label = np.array(Label)
        model = lightgbm.LGBMRegressor()
        kf = KFold(n_splits=5, shuffle=True)
        Corr = 0
        MSE = 0
        MAE = 0
        for train_index, test_index in kf.split(Ifeature, Label):
            Train_data, Train_label = Ifeature[train_index], Label[train_index]
            Test_data, Test_label = Ifeature[test_index], Label[test_index]
            model.fit(Train_data, Train_label)
            Pre_label = model.predict(Test_data)
            MSE += np.sqrt(mean_squared_error(Test_label, Pre_label))
            Corr += np.corrcoef(Test_label, Pre_label)[1][0]
            MAE += mean_absolute_error(Test_label, Pre_label)
        Corr *= 0.2
        MSE *= 0.2
        MAE *= 0.2
        print('MSE:', MSE, 'MAE:', MAE, 'Corr:', Corr)
        # data = Peptide_data.reshape((5, 3)).T
        # res = pd.DataFrame({"Corr:": data[0], "MSE": data[1], "MAE": data[2]})
        # res.to_excel('Variants_prediction_cutoff_3.xlsx')


if __name__ == '__main__':
    Features_pp, Input, Label = Load_data()
    features_normalize, Handcrafted = ALL_features(Input)
    features = (Features_pp, features_normalize, Handcrafted,
                np.concatenate((Features_pp, features_normalize), axis=1),
                np.concatenate((Features_pp, Handcrafted), axis=1),
                np.concatenate((features_normalize, Handcrafted), axis=1),
                np.concatenate((Features_pp, features_normalize, Handcrafted), axis=1))
    Validations(features, Label)
