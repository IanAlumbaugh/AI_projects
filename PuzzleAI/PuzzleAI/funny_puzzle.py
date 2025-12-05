import heapq

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    DONE: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    distance = 0
    for i in range(9):
        # Skip empty tiles
        if from_state[i] != 0:
            # Find where tile should be
            target_index = to_state.index(from_state[i])
            # Convert index to row, col coordinates
            current_row, current_col = divmod(i, 3)
            target_row, target_col = divmod(target_index, 3)
            # Get manhattan distance
            distance += abs(current_row - target_row) + abs(current_col - target_col)

    return distance
    
def get_count_heuristic(from_state, to_state=(1,2,3,4,5,6,7,0,0)):
    """
    DONE: Implement this function. This function will not directly be tested by the grader.
    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that returns the count of the number of incorrectly placed elements in the from_state variable
    """
    num_incorrect_count = 0
    for i in range(9):
        if from_state[i] != to_state[i]:
            num_incorrect_count += 1
    return num_incorrect_count


def print_succ(state):
    """
    DONE: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))

def print_count_succ(state):
    """
    DONE: This is based on get_count_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle along with count based heuristic 
    """
    
    succ_states = get_succ(state)
    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_count_heuristic(succ_state)))


def get_succ(state):
    """
    DONE: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    succ_states = []
    seen = set()  # Track states we've already generated to avoid duplicates
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # Find all positions where there are empty spaces
    zero_indices = [i for i, val in enumerate(state) if val == 0]

    for zero_index in zero_indices:
        # Convert index to row, col
        row, col = divmod(zero_index, 3)
        # Move tile from each direction into empty space
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            # Check if the new position is in grid
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_index = new_row * 3 + new_col
                # Only swap if moving a numbered tile
                if state[new_index] != 0:
                    # Create copy of state and perform swap
                    new_state = state[:]
                    new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
                    # Check if already seen the state
                    state_tuple = tuple(new_state)
                    if state_tuple not in seen:
                        seen.add(state_tuple)
                        succ_states.append(new_state)

    return sorted(succ_states)

    

def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    DONE: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    visited = set()  # Track visited states to avoid revisiting
    pq = []
    all_states = []
    parent_map = {0: -1}
    max_length = 0

    # Init priority queue
    h = get_manhattan_distance(state, goal_state)
    heapq.heappush(pq, (h, state, (0, h, 0)))
    all_states.append(state)

    while pq:
        # Track max queue length before popping
        max_length = max(max_length, len(pq))
        
        # Pop state with lowest g + h
        cost, current_state, (g, h, current_index) = heapq.heappop(pq)
        state_tuple = tuple(current_state)
        
        # Skip if already visited
        if state_tuple in visited:
            continue
        visited.add(state_tuple)

        # Check if reached goal
        if current_state == goal_state:
            # Make path from goal to start again
            path = []
            idx = current_index
            while idx != -1:
                s = all_states[idx]
                h_val = get_manhattan_distance(s, goal_state)
                path.append((s, h_val))
                idx = parent_map[idx]
            path.reverse()

            # Get state info with move counts
            state_info_list = [(s, h, i) for i, (s, h) in enumerate(path)]
            break

        # Get successors
        for successor in get_succ(current_state):
            successor_tuple = tuple(successor)
            if successor_tuple not in visited:
                # Calculate costs for successor
                h_new = get_manhattan_distance(successor, goal_state)
                g_new = g + 1
                cost_new = g_new + h_new
                
                # Add successor to all_states and note its parent
                successor_index = len(all_states)
                all_states.append(successor)
                parent_map[successor_index] = current_index
                
                # Push successor onto priority queue
                heapq.heappush(pq, (cost_new, successor, (g_new, h_new, successor_index)))

    # This is a format helper，which is only designed for format purpose.
    # build "state_info_list", for each "state_info" in the list, it contains "current_state", "h" and "move".
    # define and compute max length
    # it can help to avoid any potential format issue.
    for state_info in state_info_list:
        current_state = state_info[0]
        h = state_info[1]
        move = state_info[2]
        print(current_state, "h={}".format(h), "moves: {}".format(move))
    print("Max queue length: {}".format(max_length))



if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    # print_succ([2,5,1,4,0,6,7,0,3])
    # print()
    # print_count_succ([2,5,1,4,0,6,7,0,3])
    # print()
    # print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    # print()
    # solve([2,5,1,4,0,6,7,0,3])
    solve([4,3,0,5,1,6,7,2,0])
    print()