# Ben Kabongo
#
# Tests


import numpy as np
import matplotlib.pyplot as plt

from players import RandomPlayer, MonteCarloPlayer, UCTPlayer
from game_state import GameState


def do_stats(title, player1, player2, height=6, width=7, nb_parts=1000):
    """
    Lance une partie entre 2 joueurs
    Etudie le nombre de coup jusqu'à une victoire, pour le joueur 1 et pour le joueur 2

    :param title: titre
    :param player1, player2: joueurs
    :param height: nombre de lignes du plateau
    :param width: nombre de colonnes du plateau
    :param nb_parts: nombre de parties à lancer
    """
    tokens = height*width
    game = GameState(height, width, player1, player2)

    # tableau des résultats
    # 0 pour les parties nulles # 1 le joueur 1 # 2 pour les parties gagnées par le joueur 2
    result = np.zeros((3, tokens), dtype=int)
    for i in range(nb_parts):
        winner_value = game.run()
        if   winner_value == player1.value: result[1, game.tokens-1] += 1
        elif winner_value == player2.value: result[2, game.tokens-1] += 1
        else: result[0, game.tokens-1] += 1
        game.reset()
    # différences des victoires des joueurs
    diff = result[1] - result[2]

    print(title)
    print("Nb parties nulles\t:", result[0].sum())
    print("Nb parties player 1\t:", result[1].sum())
    print("Nb parties player 2\t:", result[2].sum())
    print("Différence \t:", diff.sum())

    # histogrammes
    bins = tokens
    title_file = title.lower().replace(" ", "_")

    # Joueur 1
    plt.title(f"{title}. Nb parties={nb_parts}. Joueur 1")
    plt.hist(np.arange(tokens), weights=result[1], bins=bins, width=1, color="blue", label="Joueur 1")
    plt.ylabel("Nb victoires")
    plt.xlabel("Nb coups")
    plt.xticks(np.arange(0, tokens+1, 2))
    plt.legend()
    plt.savefig(f"../docs/{title_file}_player1.png")
    plt.clf()

    # Joueur 2
    plt.title(f"{title}. Nb parties={nb_parts}. Joueur 2")
    plt.hist(np.arange(tokens), weights=result[2], bins=bins, width=1, color="orange", label="Joueur 2")
    plt.ylabel("Nb victoires")
    plt.xlabel("Nb coups")
    plt.xticks(np.arange(0, tokens+1, 2))
    plt.legend()
    plt.savefig(f"../docs/{title_file}_player2.png")
    plt.clf()

    # Différence
    plt.title(f"{title}. Nb parties={nb_parts}. Différence")
    plt.hist(np.arange(tokens), weights=diff, bins=bins, width=1, color="red", label="Différence")
    plt.ylabel("Différence victoires")
    plt.xlabel("Nb coups")
    plt.xticks(np.arange(0, tokens+1, 2))
    plt.legend()
    plt.savefig(f"../docs/{title_file}_diff.png")
    plt.clf()

    # Les deux joueurs
    plt.title(f"{title}. Nb parties={nb_parts}")
    plt.hist(np.arange(tokens), weights=result[1], bins=bins, width=1, color="blue", label="Joueur 1", align="left")
    plt.hist(np.arange(tokens), weights=result[2], bins=bins, width=1, color="orange", label="Joueur 2", align="right")
    plt.ylabel("Nb victoires")
    plt.xlabel("Nb coups")
    plt.xticks(np.arange(0, tokens+1, 2))
    plt.legend()
    plt.savefig(f"../docs/{title_file}_players.png")
    plt.clf()


# question 1.4
def random_play(height=6, width=7, nb_parts=1_000, title="Random"):
    player1 = RandomPlayer(+1)
    player2 = RandomPlayer(-1)
    do_stats(title, player1, player2, height, width, nb_parts)


# question 2
def monte_carlo_play(height=6, width=7, counter=30, nb_parts=100):
    player1 = MonteCarloPlayer(+1, counter)
    player2 = RandomPlayer(-1)
    print("Monte Carlo vs Random...")
    do_stats(f"Monte Carlo vs Random Compteur {counter}", player1, player2, height, width, nb_parts)

    player1 = RandomPlayer(+1)
    player2 = MonteCarloPlayer(-1, counter)
    print("Random vs Monte Carlo...")
    do_stats(f"Random vs Monte Carlo Compteur {counter}", player1, player2, height, width, nb_parts)

    player1 = MonteCarloPlayer(+1, counter)
    player2 = MonteCarloPlayer(-1, counter)
    print("Monte Carlo vs Monte Carlo...")
    do_stats(f"Monte Carlo Compteur {counter}", player1, player2, height, width, nb_parts)


# question 4
def uct_play(height=6, width=7, monte_carlo_counter=30, uct_simulations=30, nb_parts=100):
    player1 = UCTPlayer(+1, 2, uct_simulations)
    player2 = RandomPlayer(-1)
    print("UCT vs Random...")
    do_stats(f"UCT vs Random. Nb simulations={uct_simulations}", player1, player2, height, width, nb_parts)

    player1 = RandomPlayer(+1)
    player2 = UCTPlayer(-1, 2, uct_simulations)
    print("Random vs UCT...")
    do_stats(f"Random vs UCT. Nb simulation={uct_simulations}", player1, player2, height, width, nb_parts)

    player1 = UCTPlayer(+1, 2, uct_simulations)
    player2 = MonteCarloPlayer(-1, monte_carlo_counter)
    print("UCT vs Monte Carlo...")
    do_stats(f"UCT vs Monte Carlo. Compteur={monte_carlo_counter}. Simulations={uct_simulations}", player1, player2, height, width, nb_parts)

    player1 = MonteCarloPlayer(+1, monte_carlo_counter)
    player2 = UCTPlayer(-1, 2, uct_simulations)
    print("Monte Carlo vs UCT...")
    do_stats(f"Monte Carlo vs UCT. Compteur={monte_carlo_counter}. Simulations={uct_simulations}", player1, player2, height, width, nb_parts)

    player1 = UCTPlayer(+1, 2, uct_simulations)
    player2 = UCTPlayer(-1, 2, uct_simulations)
    print("UCT vs UCT...")
    do_stats(f"UCT. Nb simulation={uct_simulations}", player1, player2, height, width, nb_parts)


def monte_carlo_stats():
    counters = np.arange(1, 101, 10)
    result = np.zeros_like(counters)

    for i in range(len(counters)):
        counter = counters[i]
        player1 = MonteCarloPlayer(+1, counter)
        player2 = RandomPlayer(-1)
        game = GameState(6, 7, player1, player2)

        for k in range(100):
            winner_value = game.run()
            if winner_value == player1.value: result[i] += 1
            game.reset()

    plt.title("Monte Carlos")
    plt.plot(counters, result)
    plt.xlabel("Compteur")
    plt.ylabel("Victoires")
    plt.legend()
    plt.savefig("../docs/montecarlo_compteurs_victoires.png")
    plt.clf()


def uct_alpha_stats():
    uct_simulations = 30
    player1 = UCTPlayer(+1, 1, uct_simulations)
    player2 = UCTPlayer(-1, 2, uct_simulations)
    print("UCT vs UCT...")
    do_stats(f"UCT alpha=1 vs alpha=2. Nb simulation={uct_simulations}", player1, player2, height, width, nb_parts)

    player1 = UCTPlayer(+1, 2, uct_simulations)
    player2 = UCTPlayer(-1, 5, uct_simulations)
    print("UCT vs UCT...")
    do_stats(f"UCT alpha=2 vs alpha=5. Nb simulation={uct_simulations}", player1, player2, height, width, nb_parts)


def uct_stats():
    simulations = np.arange(1, 101, 10)
    result = np.zeros_like(simulations)

    for i in range(len(simulations)):
        simulation = simulations[i]
        player1 = UCTPlayer(+1, 2, simulation)
        player2 = RandomPlayer(-1)
        game = GameState(6, 7, player1, player2)

        for k in range(100):
            winner_value = game.run()
            if winner_value == player1.value: result[i] += 1
            game.reset()

    plt.title("UCT")
    plt.plot(counters, result)
    plt.xlabel("Compteur")
    plt.ylabel("Victoires")
    plt.legend()
    plt.savefig("../docs/uct_simulations_victoires.png")
    plt.clf()


if __name__ == '__main__':
    print("Tests...")
    print("Random vs Random..."); random_play()
    print(); monte_carlo_play()
    print("Joueurs aléatoires 4 x 4"); random_play(4, 4, 1_000, "Random4x4")
    print("Joueurs aléatoires 6 x 5"); random_play(6, 5, 1_000, "Random6x5")
    print("Joueurs aléatoires 8 x 8"); random_play(8, 8, 1_000, "Random8x8")
    print("Joueurs aléatoires 10 x 10"); random_play(10, 10, 1_000, "Random10x10")
    print("Monte Carlo stats..."); monte_carlo_stats()
    print(); uct_play()
    print("UCT alpha stats..."); uct_alpha_stats()
    print("UCT stats..."); uct_stats()
    print("Fin...")
