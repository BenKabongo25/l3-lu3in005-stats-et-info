# Ben Kabongo
#
# Grille de jeu

import matplotlib.pyplot as plt
import numpy as np


# question 1.1
def get_positions(height=6, width=7):
    """
    :param height: nombre de lignes de la grille
    :param width: nombre de colonnes de la grille
    :return la liste des quadriplets des positions gagnantes
    """
    # liste de toutes les positions gagnantes
    return np.array([
        [ [j, j + dj, j + 2*dj, j + 3*dj], [i, i + di, i + 2*di, i + 3*di] ]
        for j in range(height)
        for i in range(width)
        for di, dj in [(-1,1), (1,0), (0,1), (1,1)]
        if (0 <= i + 3 * di < width and 0 <= j + 3 * dj < height)
    ])


# question 1.2
class GameState:
    """
    état de jeu
    """

    def __init__(self,  height=6, width=7, player1=None, player2=None):
        """
        :param height: nombre de lignes de la grille
        :param width: nombre de colonnes de la grille
        :param player1: premier joueur
        :param player2: second joueur
        """
        self.height = height
        self.width = width
        self.positions = get_positions(height, width) # positions gagnantes
        self.grid = None # Initialisation de la grille de jeu
        self.tokens = 0 # nombre de jetons joués
        self.winner_value = 0
        self.player1 = player1
        self.player2 = player2
        self.current_player = player1
        # permet de retrouver la dernière colonne jouée
        self.last_column_played = None
        self.reset()

    def __eq__(self, game):
        return (self.current_player.value == game.current_player.value and
                np.all(self.grid == game.grid))

    def change_current_player(self):
        """
        change le joueur courant
        """
        if self.current_player.value == self.player1.value:
            self.current_player = self.player2
        else:
            self.current_player = self.player1

    def clone(self):
        """
        clonage de la grille courante
        :return une grille dans le même état que la grille actuelle
        """
        other = GameState(self.height, self.width, self.player1, self.player2)
        other.current_player = self.current_player
        other.grid = self.grid.copy()
        return other
    copy = clone

    def get_playables_columns(self) -> np.ndarray:
        """
        :return la liste des indices des colonnes où un joueur peut jouer
        """
        return np.array([x for x in range(self.width)
                        if np.count_nonzero(self.grid[:,x]) < self.height])

    def has_won(self) -> bool:
        """
        :return true si un joueur a gagné ; false sinon
        """
        for i in range(len(self.positions)):
            if np.all(self.grid[tuple(self.positions[i])] == self.player1.value):
                self.winner_value = self.player1.value
                return True
            if np.all(self.grid[tuple(self.positions[i])] == self.player2.value):
                self.winner_value = self.player2.value
                return True
        return False

    def is_finished(self):
        """
        :return true si un joueur a gagné ou si le plateau est plein
        """
        return self.has_won() or np.count_nonzero(self.grid) == self.grid.size

    def play(self, x:int, player):
        """
        place un jeton d'un joueur sur une colonne donnée
        :param x: numéro de la colonne où jouer
        :param player: joueur
        """
        self.grid[(self.grid[:,x] != 0).argmax()-1, x] = player.value
        self.tokens += 1
        self.last_column_played = x
        self.change_current_player()

    def reset(self):
        """
        réinitialisation du jeu
        """
        self.grid = np.zeros((self.height, self.width))
        self.tokens = 0
        self.current_player = self.player1
    reinit = reset

    def run(self) -> int:
        """
        lance la partie entre les deux joueurs
        :return 0 en cas de match nul et la valeur du joueur gagnant en cas de victoire
        """
        while not self.is_finished():
            self.current_player.play(self)
        return self.winner_value
