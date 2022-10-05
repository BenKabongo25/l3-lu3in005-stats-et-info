# Ben Kabongo
#
# Joueurs

import abc
import math
import numpy as np


class Player(abc.ABC):
    """
    classe abstraite principale des joueurs
    """
    def __init__(self, value:int):
        """
        :param value: valeur du joueur (pour les listes numpy)
        """
        self.value = value

    @abc.abstractmethod
    def do_choice(self, game) -> int:
        """
        effectue de colonnes où jouer parmi les colonnes disponibles
        méthode à redefinir dans les classes filles
        :param game: partie de jeu courante
        """
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        return self.value == other.value

    def play(self, game):
        """
        le joueur joue sur la colonne de son choix
        :param game: partie de jeu courante
        """
        self.play_column(game, self.do_choice(game))

    def play_column(self, game, x):
        """
        :param game: partie de jeu courante
        :param x: numéro de la colonne à jouer
        """
        game.play(x, self)


# question 1.3
class RandomPlayer(Player):
    """
    joueur aléatoire
    """
    def do_choice(self, game):
        return np.random.choice(game.get_playables_columns(), 1)[0]


# question 2
class MonteCarloPlayer(Player):
    """
    joueur implémentant l'algorithme de Monte Carlo
    """
    def __init__(self, value:int, counter:int=20):
        """
        :param counter: nombre de choix d'actions aléatoires
        """
        Player.__init__(self, value)
        self.counter = counter

    def do_choice(self, game):
        choices = game.get_playables_columns()
        n = len(choices)
        indices  = np.arange(n)
        nb_wins  = np.zeros(n)
        nb_parts = np.zeros(n)

        for i in range(self.counter):
            id = np.random.choice(indices)
            x = choices[id]
            nb_parts[id] += 1
            next = game.clone()
            self.play_column(next, x)
            # changement du joueur en aléatoire
            next.player1 = RandomPlayer(game.player1.value)
            next.player2 = RandomPlayer(game.player2.value)
            winner_value = next.run()
            if winner_value == self.value:
                nb_wins[id] += 1

        nb_parts[nb_parts==0] = 1
        probabilities = nb_wins / nb_parts
        return choices[np.argmax(probabilities)]


# partie 4
class UCTPlayer(Player):
    """
    Joueur implémentant UCT
    """

    class Node:
        """
        Noeud de l'arbre de jeu
        """
        def __init__(self, parent=None, action_id=0, children=[], nb_wins=0, nb_parts=0, is_explored=False, game=None):
            """
            :param parent: noeud parent
            :param action_id: numéro de la colonne associée à l'action
            :param children: les noeuds enfants
            :param nb_wins: nombre de victoires pour les simulations à ce noeuds
            :param nb_parts: nombre de parties pour les simulations à ce noeud
            :param is_explored: true si le noeud a déjà été exploré ; false sinon
            """
            self.parent     = parent
            self.action_id  = action_id
            self.children   = []
            self.nb_wins    = 0
            self.nb_parts   = 0
            self.is_explored= False

        def add_child(self, child):
            """
            ajoute un fils au noeud
            :param child: noeud enfant
            """
            if child.parent is None:
                child.parent = self
            self.children.append(child)

        def add_children_from_game(self, game):
            for action_id in game.get_playables_columns():
                child = UCTPlayer.Node(parent=self, action_id=action_id)
                self.add_child(child)

        def back_propage(self, result:int):
            """
            :param result: résultat à propager sur tout les noeuds parents
            """
            # Rétro-propagation
            parent = self
            while parent is not None:
                parent.nb_wins += result * 0.9 # coefficient qui aide à privilégier les victoires rapides
                parent.nb_parts += 1
                parent = parent.parent

        def get_child_by_action_id(self, action_id):
            """
            :param action_id: la colonne correspondant à l'action
            :return le noeud fils qui correspondant au numéro de colonne
            """
            for child in self.children:
                if child.action_id == action_id:
                    return child
            return None

        def get_unexplored_children(self):
            """
            :return la liste des noeuds fils inexplorés
            """
            return [child for child in self.children if not child.is_explored]

        def has_no_children(self):
            """
            :true si le noeud n'a pas encore d'enfants
            """
            return len(self.children) == 0

        def has_unexplored_children(self):
            """
            :return true si au moins un des fils est inexploré ; false sinon
            """
            for child in self.children:
                if not child.is_explored:
                    return True
            return False

        def select(self, alpha:int):
            """
            sélectionne le meilleur noeud avec ucb
            :param alpha: le paramètre alpha pour ucb
            """
            if len(self.children) == 0:
                return self

            nb_wins = np.array([node.nb_wins for node in self.children])
            nb_parts = np.array([node.nb_parts for node in self.children])
            nb_parts[nb_parts==0] = 1
            t = nb_parts.sum()
            probabilities = nb_wins / nb_parts
            ucb = np.sqrt(alpha * np.log(t)/nb_parts)
            probabilities_ucb = probabilities + ucb
            return self.children[np.argmax(probabilities_ucb)]


    def __init__(self, value:int, alpha:int=2, nb_simulations=20):
        """
        :param alpha: facteur multiplicatif du biais ucb
        :param nb_simulations: nombre de simulations
        :param nb_iterations: nombre d'itérations par simulation
        """
        Player.__init__(self, value)
        self.alpha = alpha
        self.nb_simulations = nb_simulations
        self.root = UCTPlayer.Node()

    def do_choice(self, game):
        # racine de l'état précédent
        root = self.root

        if root.has_no_children():
            root.add_children_from_game(game)

        # si le joueur UCT joue à la suite, la racine devient le noeud correspondant
        # à la nouvelle configuration
        if game.last_column_played is not None:
            child = root.get_child_by_action_id(game.last_column_played)
            if child is not None:
                root = child

        root.parent = None
        root.is_explored = True

        for i in range(self.nb_simulations):
            next = game.clone()
            next.player1 = RandomPlayer(game.player1.value)
            next.player2 = RandomPlayer(game.player2.value)

            # SELECTION
            node = root
            while not node.has_unexplored_children():
                # on choisit le fils avec le meilleur ucb
                child = node.select(self.alpha)
                # joueur l'action qui mène à child
                next.current_player.play_column(next, child.action_id)
                # le noeud devient child
                node = child

            # EXPANSION
            if node.has_no_children():
                node.add_children_from_game(next)

            child = np.random.choice(node.get_unexplored_children())
            child.is_explored = True
            next.current_player.play_column(next, child.action_id)
            node = child

            # SIMULATION
            winner_value = next.run()
            result = 1 if winner_value == self.value else 0

            # RETRO-PROPAGATION
            node.back_propage(result)

        self.root = root.select(self.alpha)
        self.root.parent = None

        return self.root.action_id
