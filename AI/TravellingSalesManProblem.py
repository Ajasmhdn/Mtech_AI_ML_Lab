def tsp_greedy(distance_matrix,start=0):
    n= len(distance_matrix)

    visited=[False]*n
    path=[start]
    visited[start]=True
    current_city=start

    for _ in range(n-1):
        next_city=None
        min_dist=float("inf")

        for city in range(n):
            if not visited[city] and distance_matrix[current_city][city]<min_dist:
                min_dist=distance_matrix[current_city][city]
                next_city=city
        path.append(next_city)
        visited[next_city]=True
        current_city=next_city

    path.append(start)
    return path

distance_matrix=[
    [0, 2, 9, 10],
    [1, 0, 6, 4],
    [15, 7, 0, 8],
    [6, 3, 12, 0]
]
tour=tsp_greedy(distance_matrix,start=0)
print("Tour:", tour)