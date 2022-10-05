# Ben Kabongo
#
# 05 oct 2021

import abc
import numpy as np


def tirage(uis: np.ndarray, a:int) -> int:
    """
    :param uis: paramètres de Bernouilli
    :param a: indice
    :return le tirage de Bernouilli entre 0 et 1 de paramètre d'indice a
    """
    return np.random.choice([0, 1], p=[1-uis[a], uis[a]])

def tirages_somme(uis: np.ndarray, a:int, t:int) -> int:
    """
    :param uis: paramètres de Bernouilli
    :param a: indice
    :param t: nombre de tirages
    :return la somme des t résultats des tirages de Bernouilli entre 0 et 1
        pour le paramètre d'indice a
    """
    return np.array([tirage(uis, a) for _ in range(t)]).sum()

def tirages_somme_max(uis: np.ndarray, t:int) -> int:
    """
    :param uis: paramètres de Bernouilli
    ;param t: nombre de tirages
    :return la somme des t résultats des tirages de Bernouilli entre 0 et 1
        pour le paramètre le plus grand
    """
    return tirages_somme(uis, np.argmax(uis), t)


class Algorithme(abc.ABC):
    """
    classe de base des algorithmes

    :param n     : taille (nombre d'indices)
    :param uis   : paramètres réels
    :param t     : nombre de tirages

    :param uis_a : paramètres estimées
    :param ntas_a: nombre d'occurences des indices
    :param ats_a : indices des tirages de 1 à t
    :param rts_a : résultats finaux des tirages de 1 à t
    """
    def __init__(self, uis:np.ndarray, t:int):
        self.n      = len(uis)
        self.uis    = uis
        self.t      = t
        self.ti     = 0
        self.ats_a  = np.zeros(self.t)
        self.rts_a  = np.zeros(self.t)
        self.uis_a  = np.zeros(self.n)
        self.ntas_a = np.zeros(self.n)

    @abc.abstractmethod
    def choix(self) -> int:
        """
        :return l'indice choisi
        """
        raise NotImplementedError

    def do(self):
        a = self.choix()
        self.ntas_a[a] += 1
        self.ats_a[self.ti] = a
        self.rts_a[self.ti] = tirage(self.uis, a)
        self.ti += 1

    def regret(self, tsm: int) -> int:
        """
        :param tsm: somme maximale estimée des tirages
        :return regret, différence entre la somme maximale et la somme des résultats
        """
        return tsm - self.rts_a.sum()


class AleatoireAlgorithme(Algorithme):
    """
    Algorithme aléatoire
    """
    def choix(self):
        return np.random.randint(self.n)

class GreedyAlgorithme(Algorithme):
    """
    Algorithme Greedy
    """
    def choix(self):
        if np.min(self.ntas_a) == 0:
            a = np.min(np.where(self.ntas_a == 0)[0])
        else:
            self.uis_a = np.array([sum([self.rts_a[i] for i in range(self.ti) if self.ats_a[i]==a])
                                for a in range(self.n)])/self.ntas_a
            a = np.argmax(self.uis_a)
        return a

class EGreedyAlgorithme(Algorithme):
    """
    Algorithme e-Greedy
    """
    def __init__(self, uis:np.ndarray, t:int, eps:int=0.5):
        Algorithme.__init__(self, uis, t)
        self.eps = eps

    def choix(self):
        if np.random.choice([False, True], p=[self.eps, 1-self.eps]):
            a = np.random.randint(self.n)
        else:
            if np.min(self.ntas_a) == 0:
                a = np.min(np.where(self.ntas_a == 0)[0])
            else:
                self.uis_a = np.array([sum([self.rts_a[i] for i in range(self.ti) if self.ats_a[i]==a])
                                    for a in range(self.n)])/self.ntas_a
                a = np.argmax(self.uis_a)
        return a

class UCBAlgorithme(Algorithme):
    """
    Algorithme UCB
    """
    def __init__(self, uis:np.ndarray, t:int, alpha:int=2):
        Algorithme.__init__(self, uis, t)
        self.alpha = alpha

    def choix(self):
        if np.min(self.ntas_a) == 0:
            a = np.min(np.where(self.ntas_a == 0)[0])
        else:
            self.uis_a = np.array([sum([self.rts_a[i] for i in range(self.ti) if self.ats_a[i]==a])
                                for a in range(self.n)])/self.ntas_a
            ucb = np.sqrt(self.alpha * np.log(self.ti)/self.ntas_a)
            self.uis_a += ucb
            a = np.argmax(self.uis_a)
        return a
