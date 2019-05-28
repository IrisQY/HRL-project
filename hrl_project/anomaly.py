import gym
from environment import MountainCarEnv
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

env = MountainCarEnv()

def anomaly_detection(max_step = 200, num_episode = 10000, step = 80):
    state_trials = []
    step_trials = []
    pos = []
    vel = []
    for episode in range(num_episode):
        state = env.reset()
        done = False
        step_count = 0
        state_trial = []
        step_trial = []
        pos.append(state[0])
        vel.append(state[1])
        while not done:
            action = np.random.randint(3)
            next_state, _ , done = env.step(action)
            state_trial.append(next_state)
            step_trial.append(step_count)
            pos.append(next_state[0])
            vel.append(next_state[1])
            done = done or (step_count == max_step-1)
            step_count += 1

        state_trials.append(state_trial)
        step_trials.append(step_trial)

    all_experiments = pd.DataFrame({'position': pos, 'velocity': vel})

    print('finish exploring')
    anomaly_idxs = []
    for i, trial in enumerate(step_trials):
        if len(trial) < 200:
            _step = min(step, len(trial))
            anomaly_idxs.append([(i,j) for j in trial[-_step:]])
    anomaly_states = pd.DataFrame(columns = ['position','velocity'])
    count = 0
    for storage in anomaly_idxs:
        for idx in storage:
            state = state_trials[idx[0]][idx[1]]
            anomaly_states.loc[count] = [state[0],state[1]] 
            count += 1

    print('finish anomaly detecting')

    return anomaly_states, all_experiments

def k_means(anomaly_states, all_experiments, k = 6):
    # labelled = anomaly_states[anomaly_states['position'] >= 0.5]
    # labelled['cluster'] = [k-1]*labelled.shape[0]
    to_cluster = anomaly_states[anomaly_states['position'] < 0.5]
    means = [np.mean(to_cluster['position'].values), np.mean(to_cluster['velocity'].values)]
    stds = [np.std(to_cluster['position'].values), np.std(to_cluster['velocity'].values)]
    position_norm = (to_cluster['position'].values - means[0]) / stds[0]
    velocity_norm = (to_cluster['velocity'].values - means[1]) / stds[1]

    to_cluster_norm = np.zeros((to_cluster.shape[0], 2))
    to_cluster_norm[:,0] = position_norm
    to_cluster_norm[:,1] = velocity_norm

    estimator = KMeans(n_clusters=k-1)
    estimator.fit(to_cluster_norm)
    
    print('finishing finding clusters')
    cluster_centers = estimator.cluster_centers_[:,0:2]
    for i in range(len(cluster_centers)):
        cluster_centers[i][0] = cluster_centers[i][0] * stds[0] + means[0]
        cluster_centers[i][1] = cluster_centers[i][1] * stds[1] + means[1]

    centers = pd.DataFrame({'position': cluster_centers[:,0], 'velocity': cluster_centers[:,1]})

    final = all_experiments[all_experiments['position'] >= 0.5]
    final['label'] = [k-1]*final.shape[0]
    rest = all_experiments[all_experiments['position'] < 0.5]
    _pos_norm = (rest['position'].values - means[0]) / stds[0]
    _vel_norm = (rest['velocity'].values - means[1]) / stds[1]
    rest_norm = np.zeros((rest.shape[0], 2))
    rest_norm[:,0] = _pos_norm
    rest_norm[:,1] = _vel_norm

    labels = estimator.predict(rest_norm)
    rest['label'] = labels

    df = final.append(rest)

    print('finish assigning labels')

    colors = ['red', 'green', 'blue', 'orange','yellow', 'magenta', 'black', 'purple', 'brown', 'white']
    fig, ax = plt.subplots()
    for i in range(k):
        label = i
        color = colors[i%len(colors)]
        within_label = df[df['label'] == label]
        ax.scatter(within_label['position'].values, within_label['velocity'].values, c=color,label=int(label), alpha=0.7)

    ax.legend()
    plt.xlabel('Position')
    plt.ylabel('Velocity')

    cluster_path = './anomaly/clusters_' + str(k) +'/'
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)

    plt.savefig(cluster_path + 'Clusters.png')
    df.to_csv(cluster_path + 'trial_clustering.csv', index = False)
    centers.to_csv(cluster_path + 'centers.csv', index = False)

    return cluster_centers, df

def get_cluster_fn(n_clusters=6, extra_bit = False, load_path='./anomaly/clusters_', max_step = 200, num_episode = 10000, step = 80):
    if load_path is not None:
        cluster_centers = pd.read_csv(load_path+str(n_clusters)+'/centers.csv')
        df = pd.read_csv(load_path + str(n_clusters)+'/trial_clustering.csv')
    else:
        anomaly_states, all_experiments = anomaly_detection(max_step, num_episode, step)
        cluster_centers, df = k_means(anomaly_states, all_experiments, k = n_clusters)

    cluster_centers = cluster_centers.values

    clusters_one_hot = [np.zeros(n_clusters) for i in range(n_clusters)]
    for i in range(len(clusters_one_hot)):
        clusters_one_hot[i][i] = 1

    to_label = df[df['label'] < n_clusters - 1]
    data = to_label[['position','velocity']].values
    labels = to_label['label'].values
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(data, labels)

    def check_cluster(data_point, subgoal_index, original_point = None):
        if data_point[0][0] >= 0.5:
            cluster_index = n_clusters - 1
        else:
            cluster_index = neigh.predict(data_point)[0]
        return (cluster_index == subgoal_index)


    def identify_cluster(data_point, extra_bit = False):
        if data_point[0][0] >= 0.5:
            cluster_index = n_clusters - 1
        else:
            cluster_index = neigh.predict(data_point)[0]
        cluster_one_hot = np.zeros(n_clusters)
        cluster_one_hot[cluster_index] = 1
        return cluster_one_hot

    return identify_cluster, check_cluster, n_clusters, np.array(clusters_one_hot)



if __name__=='__main__':
    anomaly_states, all_experiments = anomaly_detection()
    clusters, df = k_means(anomaly_states, all_experiments)
