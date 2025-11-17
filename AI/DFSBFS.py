graph={
    "9":["7","3"],
    "7":["5","1"],
    "3":["2","4"],
    "5":[],
    "1":[],
    "2":["11"],
    "4":[],
    "11":[]
}

def dfs(graph,node):
    visited=set()
    if node not in visited:
        visited.add(node)
        print(node,end=" ")
        for neighbour in graph[node]:
            dfs(graph,neighbour)
print("The following is the Depth First Search on Graph")
dfs(graph,"9")

def bfs(graph,start):
    visited=set()
    queue=[start]
    visited.add(start)

    while queue:
        node=queue.pop(0)
        print(node,end=" ")
        for neighbour in graph[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
print("The following is the Breadth First Search on Graph")
bfs(graph,"9")       