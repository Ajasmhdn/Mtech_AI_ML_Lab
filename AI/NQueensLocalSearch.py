import random

def print_board(board):
    n = len(board)
    for r in range(n):
        line = ""
        for c in range(n):
            line += "Q " if board[c] == r else ". "
        print(line)
    print()

def conflicts(board, col, row):
    n = len(board)
    count = 0
    for c in range(n):
        if c == col:
            continue
        r = board[c]
        if r == row or abs(r - row) == abs(c - col):
            count += 1
    return count

def min_conflicts(n, max_steps=1000):
    # Step 1: Start with a random board
    board = [random.randint(0, n - 1) for _ in range(n)]

    for step in range(max_steps):
        # Find conflicted queens
        conflicted = []
        for col in range(n):
            if conflicts(board, col, board[col]) > 0:
                conflicted.append(col)

        # If no conflicts, solution found
        if not conflicted:
            return board

        # Pick a random conflicted queen
        col = random.choice(conflicted)

        # Find row with minimum conflicts
        min_conf = n
        best_rows = []
        for row in range(n):
            c = conflicts(board, col, row)
            if c < min_conf:
                min_conf = c
                best_rows = [row]
            elif c == min_conf:
                best_rows.append(row)

        # Move queen to the best row (random tie-break)
        board[col] = random.choice(best_rows)

    return None  # No solution found within limit

# ---- Main Program ----
n = int(input("Enter the number of queens: "))
solution = min_conflicts(n)

if solution:
    print("\nSolution found:")
    print_board(solution)
else:
    print("No solution found within limit.")