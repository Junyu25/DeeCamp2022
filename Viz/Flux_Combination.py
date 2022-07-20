import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

Mydic = {'FKH1': 0, 'GCN5': 1, 'MED4': 2, 'OPI1': 3, 'RFX1': 4,
         'RGR1': 5, 'RPD3': 6, 'SPT3': 7, 'TFC7': 8, 'YAP6': 9}

def Load_data():
    data = np.array(pd.read_excel('combination_demo.xlsx'))
    Myinput = np.zeros([175, 10], dtype=float)
    Label = np.zeros([175], dtype=float)
    for i in range(len(data)):
        for j in range(3):
            if data[i][j] in Mydic.keys():
                Myinput[i][Mydic[data[i][j]]] = 1
        Label[i] = data[i][-1]
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2)
    result = tsne.fit_transform(Myinput)
    print(result.shape)
    return result, Label


# Plot t-SNE
def plot_embedding(data, Label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    plot_data_x = np.zeros([len(data)], dtype=float)
    plot_data_y = np.zeros([len(data)], dtype=float)
    for i in range(len(data)):
        plot_data_x[i] = data[i][0]
        plot_data_y[i] = data[i][1]
    plt.figure()
    plt.scatter(plot_data_x, plot_data_y, marker='o', c=Label, s=50)
    plt.title(title)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    result, Label = Load_data()
    plot_embedding(result, Label, 'T-SNE embedding of combinations')

