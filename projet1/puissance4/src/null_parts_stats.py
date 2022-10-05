# Ben Kabongo
#
# 14/10/2021
# Expériences pour les parties nulles

import numpy as np
import matplotlib.pyplot as plt

from game_state import GameState
from players import RandomPlayer


def do_stats(height=6, width=7, nb_parts=1000):
    player1 = RandomPlayer(+1)
    player2 = RandomPlayer(-1)
    game = GameState(height, width, player1, player2)

    nb_not_null = 0
    probabilities = np.zeros(nb_parts)

    for i in range(nb_parts):
        current_player = player1
        p = 1

        while not game.is_finished():
            next = game.clone()
            n = len(game.get_playables_columns())
            x = current_player.do_choice(next)
            current_player.play_column(next, x)

            if next.is_finished() and next.winner_value != 0:
                xs  = game.get_playables_columns()

                xs0 = list()
                for xi in xs:
                    next0 = game.clone()
                    current_player.play_column(next0, xi)
                    if not next0.is_finished() or next0.winner_value == 0:
                        xs0.append(xi)

                if len(xs0) == 0:
                    nb_not_null += 1
                    p = 0
                    game.reset()
                    break

                x = np.random.choice(xs0)

            p /= n
            current_player.play_column(game, x)

        probabilities[i] = p
        game.reset()

    print("Nombre de parties non nulles :", nb_not_null)

    probabilities = probabilities[probabilities!=0]
    print("Probabilité moyenne :", probabilities.mean())

    d = nb_parts - nb_not_null
    plt.title(f"Probabilités des parties nulles. {d}/{nb_parts}")
    plt.hist(np.arange(d), weights=probabilities, bins=d, label="Probabilité")
    plt.xlabel("Parties")
    plt.ylabel("Probabilités")
    plt.legend()
    plt.savefig("../docs/probabilites_parties_nulles")

if __name__ == '__main__':
    do_stats()
