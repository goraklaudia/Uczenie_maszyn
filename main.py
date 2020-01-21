from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from imblearn.under_sampling import CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, \
    TomekLinks, RandomUnderSampler
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from scipy.stats import wilcoxon
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

colorL = ListedColormap(['#F7CD7F','#D9FF8C','#B5B5B5'])
colorB = ListedColormap(['#D99007','#82B41F','#11114e'])


# 1 ---------- Letter Recognition
# dataset = pd.read_csv('data/letter.txt')
# X_data = dataset.iloc[:, 1:].values
# y_data = dataset.iloc[:, 0].values

# 2 ---------- AVILA
# dataset = pd.read_csv('data/avila.txt')
# X_data = dataset.iloc[:, :9].values
# y_data = dataset.iloc[:, 10].values

# 3 ---------- MAGIC Gamma Telescope
# dataset = pd.read_csv('data/magic.txt')
# X_data = dataset.iloc[:, :9].values
# y_data = dataset.iloc[:, 10].values

# 4 ---------- WINE
wine = datasets.load_wine()
X_data = wine.data[:, :2]
y_data = wine.target

# 5 ---------- IRIS
# iris = datasets.load_iris()
# X_data = iris.data[:, :2]
# y_data = iris.target

# 6 ---------- Liver Disorders
# dataset = pd.read_csv('data/liver.txt')
# X_data = dataset.iloc[:, :5].values
# y_data = dataset.iloc[:, 6].values

# # # 8 ---------- Glass Identification
# dataset = pd.read_csv('data/glass.txt')
# X_data = dataset.iloc[:, 1:9].values
# y_data = dataset.iloc[:, 10].values

# ---------- ABALONE -----
# dataset = pd.read_csv('data/abalone.txt')
# X_data = dataset.iloc[:, 0:].values
# y_data = dataset.iloc[:, 8].values

print(X_data.shape)
print('-------')

# ------- CNN --------
cnn = CondensedNearestNeighbour()
X_cnn, y_cnn = cnn.fit_resample(X_data, y_data)
print(X_cnn.shape)

# ------- ENN --------
enn = EditedNearestNeighbours()
X_enn, y_enn = enn.fit_resample(X_data, y_data)
print(X_enn.shape)

# ------- RENN --------
renn = RepeatedEditedNearestNeighbours()
X_renn, y_renn = renn.fit_resample(X_data, y_data)
print(X_renn.shape)

# ------- Tomek --------
tl = TomekLinks()
X_t, y_t = tl.fit_resample(X_data, y_data)
print(X_t.shape)

# ------- RUS --------
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_data, y_data)
print(X_rus.shape)

print('\n\n')


datasets = [
    {
        "X": X_data,
        "y": y_data,
        "name": "KNN"
    },
    {
        "X": X_cnn,
        "y": y_cnn,
        "name": "Condensed Nearest Neighbour"
    },
    {
        "X": X_enn,
        "y": y_enn,
        "name": "Edited Nearest Neighbours"
    },
    {
        "X": X_renn,
        "y": y_renn,
        "name": "Repeated Edited Nearest Neighbours"
    },
    {
        "X": X_t,
        "y": y_t,
        "name": "Tomek Links"
    },
    {
        "X": X_rus,
        "y": y_rus,
        "name": "Random Under Sampler"
    }
]


n_neighbors = 5
# Create color maps

res = np.zeros((len(datasets), 6))
for i, data in enumerate(datasets):

    X = data["X"]
    y = data["y"]

    kf = KFold(n_splits=5, random_state=10, shuffle = True)

    scores = np.zeros(5)
    for f, (train, test) in enumerate(kf.split(X)):
        scaler = StandardScaler()
        scaler.fit(X[train])
        X_train = scaler.transform(X[train])
        X_test = scaler.transform(X[test])

        start_time = datetime.now()
        clf = KNeighborsClassifier(n_neighbors, weights='distance', metric_params=None)
        clf.fit(X_train, y[train])
        y_pred = clf.predict(X_test)
        elapsed_time = datetime.now() - start_time
        score = accuracy_score(y[test], y_pred)
        res[f, i] = "%.3f" % score

    print("______  ", data["name"], "  ______")
    print("Classification accuracy: %.3f" % accuracy_score(y[test], y_pred))
    print("Time: ", elapsed_time)
    print("Degree of condensation: %.3f" % ((X_data.shape[0]-X.shape[0])/X_data.shape[0])*1)
    print('\n')

# --------------------- Plot dla zbiorów Iris i Wine ------------
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#
#     x_np, y_np = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
#     clf.fit(X, y)
#
#     Z = clf.predict(np.c_[x_np.ravel(), y_np.ravel()])
#     Z = Z.reshape(x_np.shape)
#
#     plt.figure()
#     plt.title("k = %i, alg = '%s'" % (n_neighbors, data["name"]))
#     plt.pcolormesh(x_np, y_np, Z, cmap=colorL)
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=colorB)
#     plt.xlim(x_np.min(), x_np.max())
#     plt.ylim(y_np.min(), y_np.max())
#
# plt.show()
# ------------------------ KONIEC WYKRESOW ----------------------


# ------------------------ WILCOXON ------------------------------
# print("--------- ACCURACY ----------")
# print(res,"\n")
#
# alpha = 0.05
# mean_scores = np.mean(res, axis=1)
# results = np.full((len(res), len(res)),False,dtype=bool)
# results_p = np.ones((len(res), len(res)))
#
# for i in range(len(res)-1):
#     for j in range(i+1,len(res)):
#         p = wilcoxon(res[i],res[j]).pvalue
#         results[i, j] =(p <= alpha and (mean_scores[i] < mean_scores[j]))
#
#         p2 = wilcoxon(res[j],res[i]).pvalue
#         results[j, i] =(p2 <= alpha and (mean_scores[j] < mean_scores[i]))
#         results_p[i,j] = "%.3f" % p
#         results_p[j,i] = "%.3f" % p2
#
# print(results,"\n")
# print(results_p)



