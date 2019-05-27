import gym
from environment import MountainCarEnv
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import preprocessing


env = MountainCarEnv()

def gaussianDataCluster(numData = 200*10000, numSigs = 4, numClusters = 6):
    # numData: number of samples generated for k-mean
    # numSigs: whole range = numSigs * xSig * 2
    xMean = (env.min_position + env.max_position) / 2
    xSig = (env.max_position - env.min_position) / 2 / numSigs
    yMean = 0
    ySig = env.max_speed / numSigs

    data = np.zeros((2,numData))
    data[0, :] = np.random.randn(numData) * xSig + xMean
    data[1, :] = np.random.randn(numData) * ySig
    maskX = np.logical_and(data[0, :] < env.max_position, data[0, :] > env.min_position) 
    maskY = np.logical_and(data[1, :] < env.max_speed, data[1, :] > -env.max_speed)
    data = data[:, np.logical_and(maskX, maskY)].T

    dataToClustered = data[data[:, 0] < 0.5, :]

    # Standardization:
    scaler = preprocessing.StandardScaler().fit(dataToClustered)
    dataScaled = scaler.transform(dataToClustered)

    estimator = KMeans(n_clusters= numClusters - 1)
    estimator.fit(dataScaled)

    cluster_centers = scaler.inverse_transform(estimator.cluster_centers_[:,0:2])
    centers = pd.DataFrame({'position': cluster_centers[:,0], 'velocity': cluster_centers[:,1]})

    df = pd.DataFrame({'position': dataToClustered[:, 0], 'velocity': dataToClustered[:, 1]})
    df['label'] = estimator.predict(dataScaled)

    rest = pd.DataFrame({'position': data[data[:, 0] >= 0.5, 0], 'velocity': data[data[:, 0] >= 0.5, 1]})
    rest['label'] = np.ones(np.sum(data[:, 0] >= 0.5)) * (numClusters - 1)

    df = df.append(rest)

    colors = ['red', 'green', 'blue', 'orange','yellow', 'magenta', 'black', 'purple', 'brown', 'white']
    fig, ax = plt.subplots()
    for i in range(numClusters):
        label = i
        color = colors[i%len(colors)]
        within_label = df[df['label'] == label]
        ax.scatter(within_label['position'].values, within_label['velocity'].values, c=color,label=int(label), alpha=0.7)

    ax.legend()
    plt.xlabel('Position')
    plt.ylabel('Velocity')

    cluster_path = './gaussian/clusters_' + str(numClusters) +'/'
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)

    plt.savefig(cluster_path + 'Clusters.png')
    df.to_csv(cluster_path + 'trial_clustering.csv', index = False)
    centers.to_csv(cluster_path + 'centers.csv', index = False)

    return cluster_centers, df


if __name__=='__main__':
    clusters, df = gaussianDataCluster()
