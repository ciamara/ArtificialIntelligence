from exceptions import AgentException
from connect4 import Connect4


class AlphaBetaAgent:
    def __init__(self, my_token='o', depth=4):
        self.my_token = my_token
        self.depth = depth

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')

        best_move = None
        best_value = float('-inf')
        alpha = float('-inf') #worst case for maximizing player(AI)
        beta = float('inf')#worst case for minimizing player(player or other agent)

        #simulates possible moves
        for move in connect4.possible_drops():
            new_game = self.simulate_move(connect4, move)
            value = self.alphabeta(new_game, self.depth - 1, False, alpha, beta)

            if value > best_value:
                best_value = value
                best_move = move

            #alpha -> best value found so far(for pruning)
            alpha = max(alpha, best_value)

        return best_move

    #minmax with pruning unnecessary branches
    def alphabeta(self, connect4, depth, maximizing, alpha, beta):

        if connect4.game_over or depth == 0:
            return self.heuristic(connect4)

        #AI alphabeta agent
        if maximizing:
            best_value = float('-inf')

            #simulates moves
            for move in connect4.possible_drops():
                new_game = self.simulate_move(connect4, move)
                value = self.alphabeta(new_game, depth - 1, False, alpha, beta)
                best_value = max(best_value, value)
                #updates alpha with best value
                alpha = max(alpha, best_value)

                #pruning -> opponent wont allow this move (save computation time)
                if beta <= alpha:
                    break  # Prune
            return best_value
        #player or other agent
        else:
            best_value = float('inf')
            #simulates moves
            for move in connect4.possible_drops():
                new_game = self.simulate_move(connect4, move)
                value = self.alphabeta(new_game, depth - 1, True, alpha, beta)
                best_value = min(best_value, value)
                #updates beta for minimizing
                beta = min(beta, best_value)
                #pruning -> opponent wont allow this move (save computation time)
                if beta <= alpha:
                    break  # Prune
            return best_value

    #creates a new game and makes the move
    def simulate_move(self, connect4, move):
        new_game = Connect4(width=connect4.width, height=connect4.height)
        new_game.board = [row[:] for row in connect4.board]
        new_game.who_moves = connect4.who_moves
        new_game.drop_token(move)
        return new_game

    def heuristic(self, connect4):
        if connect4.wins == self.my_token:
            return 1
        elif connect4.wins is not None:
            return -1
        else:
            center_count = connect4.center_column().count(self.my_token)
            return 0.2 * center_count  #center advantageous
