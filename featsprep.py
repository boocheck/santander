import numpy as np
import santanderenv as senv

def skip_duplicated_features(arr):
    feat_num = arr.shape[1]
    cols = range(feat_num)
    for i in xrange(feat_num):
        for j in xrange(i+1, feat_num):
            if cols[j] == j:
                if np.array_equiv(arr[:, i], arr[:, j]):
                    cols[j] = i

    indices = np.array([False]*feat_num)
    indices[np.unique(np.array(cols))] = True
    return indices

def review_data(arr):
    feat_num = arr.shape[1]
    stds = np.std(arr, 0).tolist()
    means = np.mean(arr, 0).tolist()
    mins = np.min(arr, 0).tolist()
    maxs = np.max(arr, 0).tolist()

    uniques = [np.unique(arr[:, i], return_counts=True) for i in xrange(feat_num)]

    for i in xrange(feat_num):
        print "{:4}: min:{:4} max:{:4} mean:{:4} std:{:4}".format(i, mins[i], maxs[i], means[i], stds[i])
        ind_sorted = np.argsort(uniques[i][1])[::-1][:min(10, uniques[i][1].size)]
        print "u-count: {:4}".format(uniques[i][0].size) + "  "  + " ".join(
            ["{:4}:{:4}".format(x[0], x[1]) for x in zip(uniques[i][0][ind_sorted].tolist(), uniques[i][1][ind_sorted].tolist())]
        )



# train_all = np.loadtxt(senv.get_config("data-train"), skiprows=1, delimiter=",")[:, :-1]
# test_all = np.loadtxt(senv.get_config("data-test"), skiprows=1, delimiter=",")
# all_all = np.concatenate((train_all, test_all))
# feat_num = all_all.shape[1]




