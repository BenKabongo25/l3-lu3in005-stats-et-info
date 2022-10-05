# Ben Kabongo
#
# Tests

import matplotlib.pyplot as plt
import numpy as np

import algorithmes


plt.style.use('seaborn-whitegrid')


def do_stats(uis:np.ndarray, t:int=1000, step_t:int=20, name=""):
    """
    :param uis: paramètres réels
    :param t: nombre de tirages max
    :param step_t: variation du nombre de tirages entre 1 et t
    :param name: nom pour les fichiers plot
    """
    n = len(uis)

    # regrets en fonction des tirages
    # 0 -> aléatoire 1 -> greedy 2 -> e_greedy 3 -> ucb
    ts = np.arange(1, t+1, step_t)
    regrets = np.zeros((4, len(ts)))

    for ti in ts:
        i = (ti-1)//step_t
        # gain max
        tsm = algorithmes.tirages_somme_max(uis, ti)

        alea = algorithmes.AleatoireAlgorithme(uis, ti) # algorithme aléatoire
        greedy = algorithmes.GreedyAlgorithme(uis, ti) # algorithme Greedy
        e_greedy = algorithmes.EGreedyAlgorithme(uis, ti) # algorithme EGreedy
        ucb = algorithmes.UCBAlgorithme(uis, ti) # algorithme UCB

        for k in range(ti):
            alea.do()
            greedy.do()
            e_greedy.do()
            ucb.do()

        regrets[0, i] = alea.regret(tsm)
        regrets[1, i] = greedy.regret(tsm)
        regrets[2, i] = e_greedy.regret(tsm)
        regrets[3, i] = ucb.regret(tsm)

    # Etude de la distribution

    plt.title(f"Distribution. N={n}")
    plt.bar(np.arange(n), uis)
    plt.savefig(f"../docs/{name}_distribution_t{t}_n{n}")
    plt.clf()

    # Etude du regret --

    plt.title(f"Aléatoire. Nombre de tirages={t}. Taille={n}")
    plt.plot(ts, regrets[0], color="blue", label="Aléatoire")
    plt.xlabel("Nombre de tirages")
    plt.ylabel("Regrets")
    plt.legend()
    plt.savefig(f"../docs/{name}_aleatoire_t{t}_n{n}")
    plt.clf()

    plt.title(f"Greedy. Nombre de tirages={t}. Taille={n}")
    plt.plot(ts, regrets[1], color="red", label="Greedy")
    plt.xlabel("Nombre de tirages")
    plt.ylabel("Regrets")
    plt.legend()
    plt.savefig(f"../docs/{name}_greedy_t{t}_n{n}")
    plt.clf()

    plt.title(f"EGreedy. Nombre de tirages={t}. Taille={n}")
    plt.plot(ts, regrets[2], color="green", label="EGreedy")
    plt.xlabel("Nombre de tirages")
    plt.ylabel("Regrets")
    plt.legend()
    plt.savefig(f"../docs/{name}_e_greedy_t{t}_n{n}")
    plt.clf()

    plt.title(f"UCB. Nombre de tirages={t}. Taille={n}")
    plt.plot(ts, regrets[3], color="yellow", label="UCB")
    plt.xlabel("Nombre de tirages")
    plt.ylabel("Regrets")
    plt.legend()
    plt.savefig(f"../docs/{name}_ucb_t{t}_n{n}")
    plt.clf()

    plt.title(f"Nombre de tirages={t}. Taille={n}")
    plt.plot(ts, regrets[0], color="blue", label="Aléatoire")
    plt.plot(ts, regrets[1], color="red", label="Greedy")
    plt.plot(ts, regrets[2], color="green", label="EGreedy")
    plt.plot(ts, regrets[3], color="yellow", label="UCB")
    plt.xlabel("Nombre de tirages")
    plt.ylabel("Regrets")
    plt.legend()
    plt.savefig(f"../docs/{name}_bandits_manchots_t{t}_n{n}")
    plt.clf()
    #plt.show()


def eps_greedy_stats(uis:np.ndarray, t:int, step_t:int):
    """
    Statistiques pour eps greedy en modifiant le paramètre eps

    :param uis: paramètres réels
    :param t: nombre de tirages max
    :param step_t: variation du nombre de tirages entre 1 et t
    """
    n = len(uis)

    # regrets en fonction des tirages
    ts = np.arange(1, t+1, step_t)
    regrets = np.zeros((5, len(ts)))

    for ti in ts:
        i = (ti-1)//step_t
        # gain max
        tsm = algorithmes.tirages_somme_max(uis, ti)

        e00 = algorithmes.EGreedyAlgorithme(uis, ti, 0.0)
        e02 = algorithmes.EGreedyAlgorithme(uis, ti, 0.2)
        e05 = algorithmes.EGreedyAlgorithme(uis, ti, 0.5)
        e08 = algorithmes.EGreedyAlgorithme(uis, ti, 0.8)
        e10 = algorithmes.EGreedyAlgorithme(uis, ti, 1.0)

        for k in range(ti):
            e00.do()
            e02.do()
            e05.do()
            e08.do()
            e10.do()

        regrets[0, i] = e00.regret(tsm)
        regrets[1, i] = e02.regret(tsm)
        regrets[2, i] = e05.regret(tsm)
        regrets[3, i] = e08.regret(tsm)
        regrets[4, i] = e10.regret(tsm)

    # Etude de la distribution

    plt.title(f"Distribution. N={n}")
    plt.bar(np.arange(n), uis)
    plt.savefig(f"../docs/eps_greedy/distribution_t{t}_n{n}")
    plt.clf()

    # Etude du regret --

    plt.title(f"Eps Greedy eps=0.0. Nombre de tirages={t}. Taille={n}")
    plt.plot(ts, regrets[0], color="blue", label="eps=0.0")
    plt.xlabel("Nombre de tirages")
    plt.ylabel("Regrets")
    plt.legend()
    plt.savefig(f"../docs/eps_greedy/eps_00_t{t}_n{n}")
    plt.clf()

    plt.title(f"Eps Greedy eps=0.2. Nombre de tirages={t}. Taille={n}")
    plt.plot(ts, regrets[1], color="red", label="eps=0.2")
    plt.xlabel("Nombre de tirages")
    plt.ylabel("Regrets")
    plt.legend()
    plt.savefig(f"../docs/eps_greedy/eps_02_t{t}_n{n}")
    plt.clf()

    plt.title(f"Eps Greedy eps=0.5. Nombre de tirages={t}. Taille={n}")
    plt.plot(ts, regrets[2], color="green", label="eps=0.5")
    plt.xlabel("Nombre de tirages")
    plt.ylabel("Regrets")
    plt.legend()
    plt.savefig(f"../docs/eps_greedy/eps_05_t{t}_n{n}")
    plt.clf()

    plt.title(f"Eps Greedy eps=0.8. Nombre de tirages={t}. Taille={n}")
    plt.plot(ts, regrets[3], color="yellow", label="eps=0.8")
    plt.xlabel("Nombre de tirages")
    plt.ylabel("Regrets")
    plt.legend()
    plt.savefig(f"../docs/eps_greedy/eps_08_t{t}_n{n}")
    plt.clf()

    plt.title(f"Eps Greedy eps=1.0. Nombre de tirages={t}. Taille={n}")
    plt.plot(ts, regrets[4], color="black", label="eps=1.0")
    plt.xlabel("Nombre de tirages")
    plt.ylabel("Regrets")
    plt.legend()
    plt.savefig(f"../docs/eps_greedy/eps_10_t{t}_n{n}")
    plt.clf()

    plt.title(f"E Greedy. Nombre de tirages={t}. Taille={n}")
    plt.plot(ts, regrets[0], color="blue", label="eps=0.0")
    plt.plot(ts, regrets[1], color="red", label="eps=0.2")
    plt.plot(ts, regrets[2], color="green", label="eps=0.5")
    plt.plot(ts, regrets[3], color="yellow", label="eps=0.8")
    plt.plot(ts, regrets[4], color="black", label="eps=1.0")
    plt.xlabel("Nombre de tirages")
    plt.ylabel("Regrets")
    plt.legend()
    plt.savefig(f"../docs/eps_greedy/eps_t{t}_n{n}")
    plt.clf()
    #plt.show()


def ucb_stats(uis:np.ndarray, t:int, step_t:int):
    """
    Statistiques pour ucb en modifiant le paramètre alpha

    :param uis: paramètres réels
    :param t: nombre de tirages max
    :param step_t: variation du nombre de tirages entre 1 et t
    """
    n = len(uis)

    # regrets en fonction des tirages
    ts = np.arange(1, t+1, step_t)
    regrets = np.zeros((3, len(ts)))

    for ti in ts:
        i = (ti-1)//step_t
        # gain max
        tsm = algorithmes.tirages_somme_max(uis, ti)

        ucb00 = algorithmes.UCBAlgorithme(uis, ti, 0.0)
        ucb10 = algorithmes.UCBAlgorithme(uis, ti, 1.0)
        ucb20 = algorithmes.UCBAlgorithme(uis, ti, 2.0)

        for k in range(ti):
            ucb00.do()
            ucb10.do()
            ucb20.do()

        regrets[0, i] = ucb00.regret(tsm)
        regrets[1, i] = ucb10.regret(tsm)
        regrets[2, i] = ucb20.regret(tsm)

    # Etude de la distribution

    plt.title(f"Distribution. N={n}")
    plt.bar(np.arange(n), uis)
    plt.savefig(f"../docs/ucb/distribution_t{t}_n{n}")
    plt.clf()

    # Etude du regret --

    plt.title(f"UCB alpha=0.0. Nombre de tirages={t}. Taille={n}")
    plt.plot(ts, regrets[0], color="blue", label="alpha=0.0")
    plt.xlabel("Nombre de tirages")
    plt.ylabel("Regrets")
    plt.legend()
    plt.savefig(f"../docs/ucb/alpha_00_t{t}_n{n}")
    plt.clf()

    plt.title(f"UCB alpha=1.0. Nombre de tirages={t}. Taille={n}")
    plt.plot(ts, regrets[1], color="green", label="alpha=1.0")
    plt.xlabel("Nombre de tirages")
    plt.ylabel("Regrets")
    plt.legend()
    plt.savefig(f"../docs/ucb/alpha_10_t{t}_n{n}")
    plt.clf()

    plt.title(f"UCB alpha=2.0. Nombre de tirages={t}. Taille={n}")
    plt.plot(ts, regrets[2], color="black", label="alpha=2.0")
    plt.xlabel("Nombre de tirages")
    plt.ylabel("Regrets")
    plt.legend()
    plt.savefig(f"../docs/ucb/alpha_20_t{t}_n{n}")
    plt.clf()

    plt.title(f"UCB. Nombre de tirages={t}. Taille={n}")
    plt.plot(ts, regrets[0], color="blue", label="alpha=0.0")
    plt.plot(ts, regrets[1], color="red", label="alpha=1.0")
    plt.plot(ts, regrets[2], color="green", label="alpha=2.0")
    plt.xlabel("Nombre de tirages")
    plt.ylabel("Regrets")
    plt.legend()
    plt.savefig(f"../docs/ucb/ucb_t{t}_n{n}")
    plt.clf()
    #plt.show()


def main():
    # tirages aléatoires
    uis = np.random.random(3)
    print("Stats of n=3, t=10, step=1...")
    do_stats(uis, 10, 1, "random10")
    print("Stats of n=3, t=100, step=10...")
    do_stats(uis, 100, 10, "random100")

    uis = np.random.random(10)
    print("Stats of n=10, t=10, step=1...")
    do_stats(uis, 10, 1, "random10")
    print("Stats of n=10, t=100, step=10...")
    do_stats(uis, 100, 10, "random100")
    print("Stats of n=10, t=1000, step=20...")
    do_stats(uis, 1000, 20, "random1000")

    # tirages biaisés
    uis = np.zeros(10)
    uis.fill(.5)
    print("Stats of n=10, t=100, step=10, u=.5...")
    do_stats(uis, 100, 10, "all05")

    uis = np.random.random(10)
    uis[uis<0.5] = 0.001
    uis[uis>=0.5] = 0.999
    print("Stats of n=10, t=100, step=10, 0.001 0.999...")
    do_stats(uis, 100, 10, "inf05set00001sup05set0999")

    uis = np.zeros(10)
    uis[0] = 1.
    print("Stats of n=10, t=100, step=10, one=1, other=0")
    do_stats(uis, 100, 10, "one1other0")

    print("Eps greedy stats...")
    uis = np.random.random(10)
    print("Stats of n=10, t=1000, step=20...")
    eps_greedy_stats(uis, 1000, 20)

    print("Ucb stats...")
    uis = np.random.random(10)
    print("Stats of n=10, t=1000, step=20...")
    ucb_stats(uis, 1000, 20)


if __name__ == '__main__':
    main()
