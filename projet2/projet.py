# Ben Kabongo B.
# Nov. 2021
# Sorbonne Université. LU3IN005


import utils
import pandas as pd
import numpy as np
import math
import scipy.stats
import matplotlib.pyplot as plt


prior = 0

def getPrior(data: pd.DataFrame) -> dict:
    """
    :param data: données
    :return: dictionnaire à trois clés:
        - estimation : estimations des lignes classifiées à 1
        - min5pourcent, max5pourcent: intervalle de confiance de 95%
    """
    N = data.shape[0]
    target = np.array(data['target'])
    estimation = np.mean(target)
    delta = 1.96 * np.std(target) / math.sqrt(N)
    global prior
    prior = int(estimation > 0.5)
    return {'estimation'  : estimation,
            'min5pourcent': estimation - delta,
            'max5pourcent': estimation + delta}


class APrioriClassifier(utils.AbstractClassifier):
    """
    classifieur à priori
    """

    def estimClass(self, attrs: dict) -> int:
        return prior

    def statsOnDF(self, data: pd.DataFrame):
        vp, fp, vn, fn = 0, 0, 0, 0
        #getPrior(data)
        for i in range(data.shape[0]):
            attrs = utils.getNthDict(data, i)
            class_ = self.estimClass(attrs)
            if attrs['target'] == 0:
                if class_ == 0: vn += 1
                else: fp += 1
            else:
                if class_ == 0: fn += 1
                else: vp += 1
        precision = vp / (vp + fp)
        rappel    = vp / (vp + fn)
        return {'VP': vp, 'VN': vn, 'FP': fp, 'FN': fn, 'précision': precision, 'rappel': rappel}


def P2D_l(data: pd.DataFrame, attr: str) -> dict:
    """
    calcule la probabilité d'un attribut sachant la valeur du target
    :param data: données
    :param attr: attribut
    :return dictionnaire des probabilités
    """
    uniques = np.sort(np.array(data[attr]))
    target0 = np.array(data.loc[data["target"] == 0][attr])
    target1 = np.array(data.loc[data["target"] == 1][attr])
    N0 = len(target0)
    N1 = len(target1)
    unique0, counts0 = np.unique(target0, return_counts=True)
    unique1, counts1 = np.unique(target1, return_counts=True)
    probabilities = dict()
    probabilities[0] = dict()
    probabilities[1] = dict()
    for i in range(len(uniques)):
        k = uniques[i]
        c0 = 0 if k not in unique0 else counts0[np.where(unique0 == k)[0]][0]
        c1 = 0 if k not in unique1 else counts1[np.where(unique1 == k)[0]][0]
        probabilities[0][k] = c0 / N0
        probabilities[1][k] = c1 / N1
    return probabilities


def P2D_p(data: pd.DataFrame, attr: str) -> dict:
    """
    calcule la probabilité du target sachant la valeur d'un attribut
    :param data: données
    :param attr: attribut
    :return dictionnaire des probabilités
    """
    uniques = np.sort(np.array(data[attr]))
    probabilities = dict()
    for i in range(len(uniques)):
        k = uniques[i]
        probabilities[k] = dict()
        d = data.loc[data[attr] == k]
        Nd = d.shape[0]
        target0 = np.array(d.loc[d["target"] == 0]["target"])
        unique0, counts0 = np.unique(target0, return_counts=True)
        target1 = np.array(d.loc[d["target"] == 1]["target"])
        unique1, counts1 = np.unique(target1, return_counts=True)
        c0 = 0 if 0 not in unique0 else counts0[np.where(unique0 == 0)[0]][0]
        c1 = 0 if 1 not in unique1 else counts1[np.where(unique1 == 1)[0]][0]
        probabilities[k][0] = c0 / Nd
        probabilities[k][1] = c1 / Nd
    return probabilities


class ML2DClassifier(APrioriClassifier):
    """
    classifieur 2D maximum de vraissemblance
    """

    def __init__(self, data: pd.DataFrame, attr: str):
        self.attr = attr
        self.probas = P2D_l(data, attr)

    def estimClass(self, attrs):
        value = attrs[self.attr]
        p_attr_t0 = self.probas[0][value]
        p_attr_t1 = self.probas[1][value]
        return int(p_attr_t1 > p_attr_t0)


class MAP2DClassifier(APrioriClassifier):
    """
    classifieur 2D maximum à postériori
    """

    def __init__(self, data: pd.DataFrame, attr: str):
        self.attr = attr
        self.probas = P2D_p(data, attr)

    def estimClass(self, attrs):
        value = attrs[self.attr]
        p_attr_t0 = self.probas[value][0]
        p_attr_t1 = self.probas[value][1]
        return int(p_attr_t1 > p_attr_t0)


def nbParams(data: pd.DataFrame, attrs: list=None) -> int:
    """
    :param data: données
    :param attrs: liste d'attributs à prendre en compte x: le nombre d'attributs
    :return le nombre d'octets en mémoire nécessaire pour la construction d'un x-classifieur
    """
    if attrs is None:
        attrs = data.keys()
    size = 8
    for attr in attrs:
        size *= len(np.unique(data[attr]))
    return size
nbrParams = nbParams


def nbParamsIndep(data: pd.DataFrame, attrs: list=None) -> int:
    """
    :param data: données
    :param attrs: liste d'attributs à prendre en compte x: le nombre d'attributs
    :return le nombre d'octets en mémoire nécessaire pour la construction d'un x-classifieur dont
            on ne stocke que les produits des probabilités
    """
    if attrs is None:
        attrs = data.keys()
    size = 0
    for attr in attrs:
        size += len(np.unique(data[attr]))
    return 8 * size
nbrParamsIndep = nbParamsIndep


def drawNaiveBayes(data: pd.DataFrame, attr: str):
    """
    rend un dessin de l'attribut en fonction du reste des attributs du dataframe
    """
    graph = ""
    attrs = data.keys()
    for attr1 in attrs:
        if attr1 == attr: continue
        graph += attr + "->" + attr1 + ";"
    return utils.drawGraph(graph)


def nbParamsNaiveBayes(data: pd.DataFrame, attr: str, attrs: list=None):
    """
    :param data: données
    :param attr: attribut en fonction duquel le calcul est fait
    :param attrs: liste d'attributs à prendre en compte
    :return le nombre d'octets en mémoire nécessaire pour un naive bayes
    """
    if attrs is None:
        attrs = data.keys()
    size = 0
    for attr1 in attrs:
        if attr1 == attr: continue
        size += len(np.unique(data[attr1]))
    u = len(np.unique(data[attr]))
    size *= u
    size += u
    return 8 * size
nbrParamsNaiveBayes = nbParamsNaiveBayes


class MLNaiveBayesClassifier(APrioriClassifier):
    """
    classifieur maximum de vraissemblance
    """

    def __init__(self, data: pd.DataFrame):
        self.probas = {attr: P2D_l(data, attr) for attr in data.keys()}

    def estimProbas(self, attrs):
        # P(attr1, attr2, ... |target) = P(attr1|target) * P(attr2|target) * ...
        p_t0, p_t1 = 1, 1
        for attr in self.probas:
            if attr == 'target': continue
            p_attr = self.probas[attr]
            p_t0 *= p_attr[0][attrs[attr]] if attrs[attr] in p_attr[0] else 0
            p_t1 *= p_attr[1][attrs[attr]] if attrs[attr] in p_attr[1] else 0
        return {0: p_t0, 1: p_t1}

    def estimClass(self, attrs):
        p = self.estimProbas(attrs)
        return int(p[1] > p[0])


class MAPNaiveBayesClassifier(APrioriClassifier):
    """
    classifieur maximum à postériori
    """

    def __init__(self, data: pd.DataFrame):
        self.probas = {attr: P2D_l(data, attr) for attr in data.keys()}
        p = getPrior(data)['estimation']
        self.p1 = p if prior == 1 else 1-p

    def estimProbas(self, attrs):
        # P(target| attr1, attr2, ...)
        p_t0, p_t1 = 1-self.p1, self.p1
        for attr in self.probas:
            if attr == 'target': continue
            p_attr = self.probas[attr]
            p_t0 *= p_attr[0][attrs[attr]] if attrs[attr] in p_attr[0] else 0
            p_t1 *= p_attr[1][attrs[attr]] if attrs[attr] in p_attr[1] else 0
        pa = p_t0 + p_t1
        if pa != 0:
            p_t0 /= pa
            p_t1 /= pa
        return {0: p_t0, 1: p_t1}

    def estimClass(self, attrs):
        p = self.estimProbas(attrs)
        return int(p[1] > p[0])


def isIndepFromTarget(data: pd.DataFrame, attr: str, alpha: float) -> bool:
    """
    :param data: données
    :param attr: attribut
    :param alpha:
    :return true si l'attribut est indépendant de target à un seuil de alpha%
    """
    # H0 : attr et target sont indépendants
    crosstab = pd.crosstab(data[attr], data['target'], margins=True, margins_name="S")
    chi2 = 0
    for j in data['target'].unique():
        for i in data[attr].unique():
            O = crosstab[j][i]
            E = crosstab[j]['S'] * crosstab['S'][i] / crosstab['S']['S']
            chi2 += ((O - E)**2)/E
    return chi2 <= scipy.stats.chi2.ppf(1-alpha, len(data[attr].unique())-1)


class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier):
    """
    classifieur à maximum de vraissemblance pour attributs non indépendants de target à partir d'un seuil
    """

    def __init__(self, data: pd.DataFrame, alpha: float):
        self.probas = {attr: P2D_l(data, attr) for attr in data.keys() if attr == 'target' or not isIndepFromTarget(data, attr, alpha)}

    def draw(self):
        graph = ""
        for attr in self.probas:
            if attr == 'target': continue
            graph += "target->" + attr + ";"
        return utils.drawGraph(graph)


class ReducedMAPNaiveBayesClassifier(MAPNaiveBayesClassifier):
    """
    classifieur à maximum à posteriori pour attributs non indépendants de target à partir d'un seuil
    """

    def __init__(self, data: pd.DataFrame, alpha: float):
        p = getPrior(data)['estimation']
        self.p1 = p if prior == 1 else 1-p
        self.probas = {attr: P2D_l(data, attr) for attr in data.keys() if attr == 'target' or not isIndepFromTarget(data, attr, alpha)}

    def draw(self):
        graph = ""
        for attr in self.probas:
            if attr == 'target': continue
            graph += "target->" + attr + ";"
        return utils.drawGraph(graph)


def mapClassifiers(dic: dict, data: pd.DataFrame):
    """
    représentation graphique des classifieurs en fonction de la précision et du rappel
    :param dic: dictionnaire des classifieurs
    :param data: données
    """
    fig = plt.figure()
    for name, classifier in dic.items():
        statsOnDF = classifier.statsOnDF(data)
        x, y = statsOnDF['précision'], statsOnDF['rappel']
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x+.005, y+.005, name, fontsize=9)
    plt.show()


def MutualInformation(data: pd.DataFrame, x: str, y: str) -> float:
    """
    :param data: données
    :parm x, y: attributs
    :return l'indice d'information mutuelle entre X et Y I(X,Y)
    """
    N = len(data)
    XY = np.array(data[[x,y]])
    X = XY[:,0]
    Y = XY[:,1]
    Uxy, Pxy = np.unique(XY, axis=0, return_counts=True)
    Ux, Px = np.unique(X, return_counts=True)
    Uy, Py = np.unique(Y, return_counts=True)
    Pxy = Pxy*1.0/N
    Px = Px*1.0/N
    Py = Py*1.0/N
    I = 0
    for [xi, yi] in Uxy:
        i = np.where((Uxy[:,0]==xi)&(Uxy[:,1]==yi))[0][0]; pxy = Pxy[i]
        i = np.where(Ux==xi)[0][0]; px = Px[i]
        i = np.where(Uy==yi)[0][0]; py = Py[i]
        if px != 0 and py != 0 and pxy != 0:
            I += pxy * np.log2((pxy)/(px*py))
    return I


def ConditionalMutualInformation(data: pd.DataFrame, x:str, y:str, z:str) -> float:
    """
    :param data: données
    :parm x, y, z: attributs
    :return l'indice d'information mutuelle contditionelle entre X et Y|Z I(X,Y|Z)
    """
    N = len(data)
    XYZ = np.array(data[[x,y,z]])
    XZ = XYZ[:,[0,2]]
    YZ = XYZ[:,[1,2]]
    Z = XYZ[:,2]
    Uxyz, Pxyz = np.unique(XYZ, axis=0, return_counts=True)
    Uxz, Pxz = np.unique(XZ, axis=0, return_counts=True)
    Uyz, Pyz = np.unique(YZ, axis=0, return_counts=True)
    Uz, Pz = np.unique(Z, return_counts=True)
    Pxyz = Pxyz*1.0/N
    Pxz = Pxz*1.0/N
    Pyz = Pyz*1.0/N
    Pz = Pz*1.0/N
    I = 0
    for [xi, yi, zi] in Uxyz:
        i = np.where((Uxyz[:,0]==xi)&(Uxyz[:,1]==yi)&(Uxyz[:,2]==zi))[0][0]; pxyz = Pxyz[i]
        i = np.where((Uxz[:,0]==xi)&(Uxz[:,1]==zi))[0][0]; pxz = Pxz[i]
        i = np.where((Uyz[:,0]==yi)&(Uyz[:,1]==zi))[0][0]; pyz = Pyz[i]
        i = np.where(Uz==zi)[0][0]; pz = Pz[i]
        if pz != 0 and pxz != 0 and pyz != 0 and pxyz != 0:
            I += pxyz * np.log2((pz*pxyz)/(pxz*pyz))
    return I


def MeanForSymetricWeights(a: np.ndarray) -> float:
    """
    :param a: matrice des informations de dépendance
    :return moyenne de la matrice
    """
    return np.sum(a) / (a.shape[0]**2-np.count_nonzero(a))


def SimplifyConditionalMutualInformationMatrix(a: np.ndarray):
    """
    :param a: matrice des informations de dépendance
    :return copie de a avec les valeurs plus petites que la moyenne ramenée à 0
    """
    a[a<np.mean(a)] = 0
simplifyContitionalMutualInformationMatrix=SimplifyConditionalMutualInformationMatrix


def Kruskal(data: pd.DataFrame, a: np.array) -> list:
    """
    ;param data: données + noms des attributs
    :param a: matrice des information de dépendance
    :return liste des attributs résultat de l'algorithme de Kruskal sur le graphe des attributs
    """
    N = len(data.keys()) - 1 # excepted target
    graph = sorted([[v1, v2, a[v1,v2]] for v1 in range(N) for v2 in range(N) if a[v1,v2]!=0],
                    key=lambda edge: edge[2],
                    reverse=True)
    M = len(graph)
    parent = {v:v for v in range(N)}
    rank = {v:0 for v in range(N)}

    def find(v):
        if parent[v] != v:
            parent[v] = find(parent[v])
        return parent[v]

    def union(v1, v2):
        root1 = find(v1)
        root2 = find(v2)
        if root1 == root2:
            return
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root1] = root2
        if rank[root1] == rank[root2]:
            rank[root2] += 1

    A = []
    for k in range(M):
        v1, v2, val = graph[k]
        if find(v1) != find(v2):
            A.append((data.keys()[v1], data.keys()[v2], val))
            union(v1, v2)
    return A


def ConnexSet(list_arcs: list) -> list:
    """
    :param list_arcs: résultat de Kruskal
    :return une forêt orientée en fonction des informations mutuelles des attributs
    """
    vs = set()
    edges = dict()
    for v1, v2, _ in list_arcs:
        vs.add(v1)
        vs.add(v2)
        if v1 not in edges: edges[v1] = set()
        if v2 not in edges: edges[v2] = set()
        edges[v1].add(v2)
        edges[v2].add(v1)
    see = {v:False for v in vs}
    def explore(v, connex):
        see[v] = True
        connex.add(v)
        for v1 in edges[v]:
            if not see[v1]:
                explore(v1, connex)
    F = []
    for v in vs:
        if not see[v]:
            connex = set()
            explore(v, connex)
            F.append(connex)
    return F
ConnexSets=ConnexSet


def OrientConnexSets(data: pd.DataFrame, list_arcs: list, class_: str) -> list:
    """
    :param data: données
    :param list_arcs: résultat de Kruskal
    :param class_: classe
    :return forêt orientée des attributs en fonction de la classe
    """
    vs = set()
    edges = dict()
    for v1, v2, _ in list_arcs:
        vs.add(v1)
        vs.add(v2)
        if v1 not in edges: edges[v1] = set()
        if v2 not in edges: edges[v2] = set()
        edges[v1].add(v2)
        edges[v2].add(v1)
    see = {v:False for v in vs}
    E = []
    def explore(v):
        see[v] = True
        connex.add(v)
        for v1 in edges[v]:
            if not see[v1]:
                E.append((v, v1))
                explore(v1)
    connexes = ConnexSet(list_arcs)
    for connex in connexes:
        list_connex = list(connex)
        mi = np.array([MutualInformation(data, v, class_) for v in list_connex])
        root = list_connex[np.argmax(mi)]
        explore(root)
    return E


class MAPTANClassifier(ReducedMAPNaiveBayesClassifier):
    """
    Classificateur TAN
    """

    def __init__(self, data: pd.DataFrame):
        p = getPrior(data)['estimation']
        self.p1 = p if prior == 1 else 1-p
        self.attrs = data.keys()
        # matrice d'informations mutuelles conditionnellement ) target
        cmis = np.array([[0 if x==y else ConditionalMutualInformation(data,x,y,"target")
                        for x in data.keys() if x!="target"]
                        for y in data.keys() if y!="target"])
        SimplifyConditionalMutualInformationMatrix(cmis)
        # forêt des attributs après Kruskal
        list_arcs = Kruskal(data, cmis)
        self.graph_edges = np.array(OrientConnexSets(data, list_arcs, 'target'))
        # probabilités des attributs en fonction de leurs parents
        # les clés du dictionnaire sont les attributs attr
        # les valeurs du dictionnaire sont des dictionnaires de probabilités
        # les clés de ces dictionnaires sont les arrays avec les valeurs de parents
        # les valeurs de ces dictionnaires sont des dictionnaires dont les clés sont
        # les valeurs uniques de attr et les valeurs les probabilités de P(attr=...|parents=...)
        self.probas = dict()
        # dictionnaire des listes des parents des attributs
        self.parents = dict()
        for attr in data.keys():
            if attr == 'target': continue
            attr_probas = dict()
            # liste des parents de l'attribut
            id = np.where(self.graph_edges[:,1]==attr)[0]
            parents = self.graph_edges[id,:][:,0]
            self.parents[attr] = parents
            # si l'attribut n'a pas de parent, son parent est target
            #if len(parents) == 0:
                # parents = np.array(['target'])
                # on sait déjà calculer P(attr[target])
            #    p_attr = P2D_l(data, attr)
            #    attr_probas[(0,)] = p_attr[0]
            #    attr_probas[(1,)] = p_attr[1]
            #    continue
            # calcul de la probailité de attr en fonction de ses parents
            keys = [attr] + list(parents) + ['target']
            # tableau de X et de ses parents
            X_PA = np.array(data[keys])
            X = X_PA[:,0]
            Ux = np.unique(X)
            PA = X_PA[:,1:]
            Upa, Cpa = np.unique(PA, return_counts=True, axis=0)
            for i in range(len(Upa)):
                pa = Upa[i]
                # P(X=x|parent=y) = nombre de x=x et parent = y/ nombre parent = y
                attr_probas[tuple(pa)] = {x : len((X_PA == np.array([x] + list(pa))).all(axis=1).nonzero()[0])/Cpa[i]
                                for x in Ux}
            self.probas[attr] = attr_probas

    def estimProbas(self, attrs):
        # P(target|attrs) = P(attrs, target)/P(attrs)
        p_t0, p_t1 = 1-self.p1, self.p1
        #p_t0, p_t1 = 1, 1
        for attr in self.probas:
            if attr == 'target': continue
            p_attr = self.probas[attr]
            key = [attrs[p] for p in self.parents[attr]]
            #print(attr, parents, key, p_attr[tuple(key + [0])][attrs[attr]], p_attr[tuple(key + [1])][attrs[attr]])
            p_t0 *= p_attr.get(tuple(key + [0]), dict()).get(attrs[attr], 0)
            p_t1 *= p_attr.get(tuple(key + [1]), dict()).get(attrs[attr], 0)
        pa = p_t0 + p_t1
        if pa != 0:
            p_t0 /= pa
            p_t1 /= pa
        return {0:p_t0, 1:p_t1}

    def draw(self):
        graph = ""
        for attr in self.attrs:
            if attr == 'target': continue
            graph += "target->" + attr + ";"
        for v1, v2 in self.graph_edges:
            graph += v1 + '->' + v2 + ';'
        return utils.drawGraph(graph)
