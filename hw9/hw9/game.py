import copy
import random
import time

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    pieces = ['b', 'r']
    max_depth = 3

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.board = [[' ' for j in range(5)] for i in range(5)]
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        """ 
        Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        # Find out what phase it is
        piece_count = sum(1 for r in range(5) for c in range(5) if state[r][c] != ' ')
        drop_phase = piece_count < 8

        # Depth-limited minimax from current state
        best_score = float('-inf')
        best_move = None

        # Generate successors for our turn
        for move, next_state in self.succ(state, self.my_piece):
            score = self.min_value(next_state, 1)
            if score > best_score:
                best_score = score
                best_move = move

        # If no successors, pick first empty or random adjacent
        if best_move is None:
            if drop_phase:
                for r in range(5):
                    for c in range(5):
                        if state[r][c] == ' ':
                            return [(r, c)]
            else:
                # Find any legal adjacent move
                for r in range(5):
                    for c in range(5):
                        if state[r][c] == self.my_piece:
                            for dr in (-1, 0, 1):
                                for dc in (-1, 0, 1):
                                    if dr == 0 and dc == 0:
                                        continue
                                    nr, nc = r + dr, c + dc
                                    if 0 <= nr < 5 and 0 <= nc < 5 and state[nr][nc] == ' ':
                                        return [(nr, nc), (r, c)]
            # As a last resort, choose a random empty
            empties = [(r, c) for r in range(5) for c in range(5) if state[r][c] == ' ']
            if empties:
                return [random.choice(empties)]

        return best_move

    def succ(self, state, my_piece): 
        """
        Generate a list of valid successors for the current game state
        on placing or moving your piece (defined by my_piece).

        Returns:
            List of tuples: [(move, next_state), ...]
            move is either [(row, col)] in drop phase or [(row, col), (src_row, src_col)] in move phase.
        """
        successors = []
        # Drop phase if total pieces < 8
        piece_count = sum(1 for r in range(5) for c in range(5) if state[r][c] != ' ')
        drop_phase = piece_count < 8

        if drop_phase:
            # Place on any empty cell
            for r in range(5):
                for c in range(5):
                    if state[r][c] == ' ':
                        ns = copy.deepcopy(state)
                        ns[r][c] = my_piece
                        successors.append(([(r, c)], ns))
        else:
            # Move any of my pieces to an adjacent empty cell (8 directions)
            for r in range(5):
                for c in range(5):
                    if state[r][c] == my_piece:
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < 5 and 0 <= nc < 5 and state[nr][nc] == ' ':
                                    ns = copy.deepcopy(state)
                                    ns[r][c] = ' '
                                    ns[nr][nc] = my_piece
                                    successors.append(([(nr, nc), (r, c)], ns))
        return successors
    
    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    
    def heuristic_game_value(self, state):
        """ 
        Define the heuristic game value of the current board state taking into account players
        and opponents

        Args:
            state (list of lists): either the current state of the game as saved in
                this TeekoPlayer object, or a generated successor state.

        Returns:
            float heuristic_val (heuristic computed for the game state)
        """
        # If there is winner, return exact value
        gv = self.game_value(state)
        if gv != 0:
            return float(gv)

        # Calculate line-based score for both players across all lines of length 4
        def line_scores(board, piece):
            score = 0
            opp = self.pieces[0] if piece == self.pieces[1] else self.pieces[1]

            # rows: 5 rows, lengths of 4
            for r in range(5):
                for c0 in range(2):
                    window = [board[r][c0 + k] for k in range(4)]
                    if opp in window:
                        continue
                    cnt = sum(1 for v in window if v == piece)
                    # more score for 3-in-4 and 2-in-4
                    if cnt == 3:
                        score += 3
                    elif cnt == 2:
                        score += 1
                    elif cnt == 4:
                        score += 100
            # cols: 5 cols, lengths of 4
            for c in range(5):
                for r0 in range(2):
                    window = [board[r0 + k][c] for k in range(4)]
                    if opp in window:
                        continue
                    cnt = sum(1 for v in window if v == piece)
                    if cnt == 3:
                        score += 3
                    elif cnt == 2:
                        score += 1
                    elif cnt == 4:
                        score += 100
            # diagonals down-right
            for r0 in range(2):
                for c0 in range(2):
                    window = [board[r0 + k][c0 + k] for k in range(4)]
                    if opp in window:
                        continue
                    cnt = sum(1 for v in window if v == piece)
                    if cnt == 3:
                        score += 3
                    elif cnt == 2:
                        score += 1
                    elif cnt == 4:
                        score += 100
            # diagonals down-left
            for r0 in range(2):
                for c0 in range(3, 5):
                    window = [board[r0 + k][c0 - k] for k in range(4)]
                    if opp in window:
                        continue
                    cnt = sum(1 for v in window if v == piece)
                    if cnt == 3:
                        score += 3
                    elif cnt == 2:
                        score += 1
                    elif cnt == 4:
                        score += 100
            # 2x2 boxes
            for r in range(4):
                for c in range(4):
                    box = [board[r + dr][c + dc] for dr in (0, 1) for dc in (0, 1)]
                    if opp in box:
                        continue
                    cnt = sum(1 for v in box if v == piece)
                    if cnt == 3:
                        score += 4
                    elif cnt == 2:
                        score += 1
                    elif cnt == 4:
                        score += 100
            return score

        my_score = line_scores(state, self.my_piece)
        opp_score = line_scores(state, self.opp)

        # Normalize using smooth scaling
        raw = my_score - opp_score
        # Limit for extreme values
        cap = 50.0
        raw = max(-cap, min(cap, raw))
        heuristic_val = raw / cap
        return heuristic_val
 
    def game_value(self, state):
        """ 
        Checks the current board status for a win condition

        Args:
            state (list of lists): either the current state of the game as saved in
                this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner
        """
        def winner_for(piece):
            # 4-in-a-row horizontally
            for r in range(5):
                for c0 in range(2):
                    if all(state[r][c0 + k] == piece for k in range(4)):
                        return True
            # 4-in-a-row vertically
            for c in range(5):
                for r0 in range(2):
                    if all(state[r0 + k][c] == piece for k in range(4)):
                        return True
            # 4-in-a-row diagonally down-right
            for r0 in range(2):
                for c0 in range(2):
                    if all(state[r0 + k][c0 + k] == piece for k in range(4)):
                        return True
            # 4-in-a-row diagonally down-left
            for r0 in range(2):
                for c0 in range(3, 5):
                    if all(state[r0 + k][c0 - k] == piece for k in range(4)):
                        return True
            # 2x2 box
            for r in range(4):
                for c in range(4):
                    if (state[r][c] == piece and state[r][c+1] == piece and
                        state[r+1][c] == piece and state[r+1][c+1] == piece):
                        return True
            return False

        if winner_for(self.my_piece):
            return 1
        if winner_for(self.opp):
            return -1
        return 0
    
    def max_value(self, state, depth):
        """
        Complete the helper function to implement min-max as described in the writeup
        """
        # Winner test
        gv = self.game_value(state)
        if gv != 0:
            return float(gv)
        if depth >= self.max_depth:
            return self.heuristic_game_value(state)

        v = float('-inf')
        # Our turn: choose best successor
        for _, next_state in self.succ(state, self.my_piece):
            v = max(v, self.min_value(next_state, depth + 1))
        return v

    def min_value(self, state, depth):
        """
        Min node for opponent's turn.
        """
        # Winner test
        gv = self.game_value(state)
        if gv != 0:
            return float(gv)
        if depth >= self.max_depth:
            return self.heuristic_game_value(state)

        v = float('inf')
        # Opponent's turn successors
        for _, next_state in self.succ(state, self.opp):
            v = min(v, self.max_value(next_state, depth + 1))
        return v


############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()