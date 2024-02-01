import numpy as np
import random
import time

class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]  # 3x3 Tic Tac Toe board
        self.current_winner = None  # Keep track of the winner!

    def print_board(self):
        for row in [self.board[i*3:(i+1)*3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')

    @staticmethod
    def print_board_nums():
        number_board = [[str(i) for i in range(j*3, (j+1)*3)] for j in range(3)]
        for row in number_board:
            print('| ' + ' | '.join(row) + ' |')

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def empty_squares(self):
        return ' ' in self.board

    def num_empty_squares(self):
        return self.board.count(' ')

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def winner(self, square, letter):
        row_ind = square // 3
        row = self.board[row_ind*3:(row_ind + 1)*3]
        if all([spot == letter for spot in row]):
            return True

        col_ind = square % 3
        column = [self.board[col_ind+i*3] for i in range(3)]
        if all([spot == letter for spot in column]):
            return True

        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all([spot == letter for spot in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all([spot == letter for spot in diagonal2]):
                return True

        return False
    
class HumanPlayer:
    def __init__(self, letter):
        self.letter = letter

    def get_move(self, game):
        valid_square = False
        val = None
        while not valid_square:
            square = input(self.letter + '\'s turn. Input move (0-8): ')
            try:
                val = int(square)
                if val not in game.available_moves():
                    raise ValueError
                valid_square = True
            except ValueError:
                print('Invalid square. Try again.')

        return val

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}  # The Q-table for storing state-action values
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def get_state(self, game):
        return str(game.board)  # Convert the board to a string to use as a state

    def get_action(self, state, available_moves):
        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            return random.choice(available_moves)
        else:
            # Exploit: choose the best action based on current Q-table
            self.q_table.setdefault(state, {})
            q_values = [self.q_table[state].get(move, 0) for move in available_moves]
            max_q_value = max(q_values)
            # In case there're several actions with the same Q-value, we randomly choose one
            actions_with_max_q_value = [move for move, q in zip(available_moves, q_values) if q == max_q_value]
            return random.choice(actions_with_max_q_value)

    def update_q_table(self, old_state, action, reward, new_state, done):
        old_q_value = self.q_table.setdefault(old_state, {}).get(action, 0)
        max_future_q = max(self.q_table.setdefault(new_state, {}).values(), default=0)
        new_q_value = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * max_future_q)
        self.q_table[old_state][action] = new_q_value

        if done:
            self.epsilon = max(self.epsilon * 0.99, 0.01)  # Decrease epsilon
    
    def get_move(self, game):
        state = self.get_state(game)
        available_moves = game.available_moves()
        return self.get_action(state, available_moves)

# Training process (simplified)
def train_agent(n_episodes=1000):
    agent = QLearningAgent()
    for episode in range(n_episodes):
        game = TicTacToe()
        state = agent.get_state(game)

        while not game.current_winner and game.empty_squares():
            action = agent.get_action(state, game.available_moves())
            game.make_move(action, 'X')  # Assume the agent always plays 'X'
            new_state = agent.get_state(game)
            reward = 1 if game.current_winner == 'X' else 0
            agent.update_q_table(state, action, reward, new_state, game.current_winner is not None)
            state = new_state

            # Opponent's move (random)
            if game.empty_squares() and not game.current_winner:
                game.make_move(random.choice(game.available_moves()), 'O')

    return agent

# Create a trained agent
trained_agent = train_agent()

def play(game, x_player, o_player, print_game=True):
    if print_game:
        game.print_board_nums()

    letter = 'X'
    while game.empty_squares():
        if letter == 'O':
            square = o_player.get_move(game)
        else:
            square = x_player.get_move(game)

        if game.make_move(square, letter):
            if print_game:
                print(letter + f' makes a move to square {square}')
                game.print_board()
                print('')

            if game.current_winner:
                if print_game:
                    print(letter + ' wins!')
                return letter

            letter = 'O' if letter == 'X' else 'X'

        time.sleep(0.8)

    if print_game:
        print('It\'s a tie!')

# Now you can use `trained_agent` to play against a human in your game setup.
human_player = HumanPlayer('X')
qlearning_agent_player = QLearningAgent('O')
tictactoe = TicTacToe()
play(tictactoe, human_player, qlearning_agent_player, print_game=True)