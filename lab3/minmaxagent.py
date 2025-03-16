from exceptions import AgentException
from connect4 import Connect4


class MinMaxAgent:
    #depth ->how many moves ahead to analyse
    def __init__(self, my_token='o', depth=4):
        self.my_token = my_token
        self.depth = depth

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        
        #move that leads to best outcome
        best_move = None
        #highest evaluation
        best_value = float('-inf')
        #evaluates each possible move using minmax
        for move in connect4.possible_drops():
            new_game = self.simulate_move(connect4, move)
            value = self.minmax(new_game, self.depth - 1, False)
            
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move
    
    #recursive minmax for evaluating moves
    def minmax(self, connect4, depth, maximizing):
        
        #if reached max depth -> heuristic(in connect4 middle positions advantageous)
        if connect4.game_over or depth == 0:
            return self.heuristic(connect4)
        
        #alternates between maximizing and minimizing score
        #AI minmax agent
        if maximizing:
            best_value = float('-inf')

            #recursive call for evaluating moves
            for move in connect4.possible_drops():
                new_game = self.simulate_move(connect4, move)
                value = self.minmax(new_game, depth - 1, False)
                best_value = max(best_value, value)
            return best_value
        #player or other agent
        else:
            best_value = float('inf')

            #recursive call for evaluating moves
            for move in connect4.possible_drops():
                new_game = self.simulate_move(connect4, move)
                value = self.minmax(new_game, depth - 1, True)
                best_value = min(best_value, value)
            #worst case for AI
            return best_value
    
    def simulate_move(self, connect4, move):

        #copies game after making a move
        new_game = Connect4(width=connect4.width, height=connect4.height)
        new_game.board = [row[:] for row in connect4.board]
        new_game.who_moves = connect4.who_moves
        #copies board and makes the move
        new_game.drop_token(move)
        #returns new game state
        return new_game
    
    #heuristic evaluation of the game, assigns score based on outcome
    def heuristic(self, connect4):
        if connect4.wins == self.my_token:
            return 1
        elif connect4.wins is not None:
            return -1
        else:
            center_count = connect4.center_column().count(self.my_token)
            return 0.2 * center_count  #center advantageous