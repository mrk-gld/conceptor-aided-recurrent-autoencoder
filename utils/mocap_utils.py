import numpy as np

def get_mocap_data(folder, dataset_names):

    data_sets = []
    data_sets_norm = []

    for names in dataset_names:
        dataset_file = folder + "data_" + names + ".npy"

        data = np.load(dataset_file)
        data_sets.append(data)

    data = np.concatenate(data_sets)

    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)

    for i in range(2):
        
        data_norm = (data_sets[i] - data_mean)/data_std
        data_sets_norm.append(data_norm)
    
    lengths = [dataset.shape[0] for dataset in data_sets]
    min_len = min(lengths)
    datasets = []
    datasets_no_norm = []
    for i in range(2):
        datasets.append(data_sets_norm[i][-min_len:])
        datasets_no_norm.append(data_sets[i][-min_len:])
        
    return np.array(datasets), np.array(datasets_no_norm), data_mean, data_std