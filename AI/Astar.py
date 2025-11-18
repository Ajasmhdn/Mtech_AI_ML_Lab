def a_star_search_simple(graph, start, goal, heuristic):
    open_set = [(start, 0, [start])]  # list of tuples: (node, cost_so_far, path)
    closed_set = set()

    while open_set:
        # Find node with lowest cost + heuristic
        current_index = 0
        current_node, current_cost, path = open_set[0]
        for i, (node, cost, p) in enumerate(open_set):
            if cost + heuristic(node) < current_cost + heuristic(current_node):
                current_index = i
                current_node, current_cost, path = node, cost, p

        open_set.pop(current_index)

        if current_node == goal:
            return path, current_cost

        if current_node in closed_set:
            continue
        closed_set.add(current_node)

        for neighbor, cost in graph.get(current_node, {}).items():
            if neighbor not in closed_set:
                new_cost = current_cost + cost
                open_set.append((neighbor, new_cost, path + [neighbor]))

    return None, float('inf')


# Example heuristic function
def example_heuristic(node):
    heuristics = {'A': 7, 'B': 6, 'C': 2, 'D': 1, 'E': 0}
    return heuristics.get(node, float('inf'))


# Example graph
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'C': 2, 'D': 5},
    'C': {'D': 1},
    'D': {'E': 3},
    'E': {}
}

path, cost = a_star_search_simple(graph, 'A', 'E', example_heuristic)
print("Path:", path)
print("Cost:", cost)