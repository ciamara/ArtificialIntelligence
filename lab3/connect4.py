from exceptions import GameplayException


class Connect4:
    def __init__(self, width=5, height=4):

        #initializes parameters
        self.width = width
        self.height = height
        self.who_moves = 'o'
        self.game_over = False
        self.wins = None

        #initializes empty board
        self.board = []
        for n_row in range(self.height):
            self.board.append(['_' for _ in range(self.width)])

    #returns list of possible drop columns(top row empty)
    def possible_drops(self):
        return [n_column for n_column in range(self.width) if self.board[0][n_column] == '_']

    #drops token onto the board
    def drop_token(self, n_column):
        if self.game_over:
            raise GameplayException('game over')
        if n_column not in self.possible_drops():
            raise GameplayException('invalid move')

        n_row = 0
        while n_row + 1 < self.height and self.board[n_row+1][n_column] == '_':
            n_row += 1
        #inserts token of current player in the lowest row of chosen column
        self.board[n_row][n_column] = self.who_moves
        #checks if game over
        self.game_over = self._check_game_over()
        #sets next player
        self.who_moves = 'o' if self.who_moves == 'x' else 'x'

    #returns middle column of the board for heuristic evaluation(center control advantageous)
    def center_column(self):
        return [self.board[n_row][self.width//2] for n_row in range(self.height)]

    #checks for wins (connected fours)
    def iter_fours(self):
        # horizontal
        for n_row in range(self.height):
            for start_column in range(self.width-3):
                yield self.board[n_row][start_column:start_column+4]

        # vertical
        for n_column in range(self.width):
            for start_row in range(self.height-3):
                yield [self.board[n_row][n_column] for n_row in range(start_row, start_row+4)]

        # diagonal
        for n_row in range(self.height-3):
            for n_column in range(self.width-3):
                yield [self.board[n_row+i][n_column+i] for i in range(4)]  # decreasing
                yield [self.board[n_row+i][self.width-1-n_column-i] for i in range(4)]  # increasing

    #checks if game is over
    def _check_game_over(self):
        #board full(tie)
        if not self.possible_drops():
            self.wins = None  # tie
            return True

        #checks for who won
        for four in self.iter_fours():
            if four == ['o', 'o', 'o', 'o']:
                self.wins = 'o'
                return True
            elif four == ['x', 'x', 'x', 'x']:
                self.wins = 'x'
                return True
        return False

    #draws onto terminal
    def draw(self):
        for row in self.board:
            print(' '.join(row))
        if self.game_over:
            print('game over')
            print('wins:', self.wins)
        else:
            print('now moves:', self.who_moves)
            print('possible drops:', self.possible_drops())