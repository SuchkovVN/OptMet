from heapq import heappop, heappush
from dataclasses import dataclass, field

@dataclass(order=True)
class Node:
    idx: int=field(compare=False)
    value: float

class Graph:
    def __init__(self):
        self.weights = []
        self.adjLists = []
        self.NumVertices = 0
        self.marks = []
        self.constant = []
    def __init__(self, weights):
        self.NumVertices = len(weights[0])
        self.weights = weights
        self.adjLists = []
        self.marks = []
        self.constant = [0 for _ in range(self.NumVertices)]
        for i in range(self.NumVertices):
            adjacent = []
            for j in range(self.NumVertices):
                if (weights[i][j] >= 0):
                    adjacent.append(j)
            self.adjLists.append(adjacent)

    def cost(self, i, j):
        return self.weights[i][j]

def Dijkstra(g, start = 0, final = 0):
    marks = [float('inf') for i in range(g.NumVertices)]
    marks[start] = 0

    prev = [0 for i in range(g.NumVertices)]
    temp = []
    v = start
    while(v != final):
        for adj in g.adjLists[v]:
            newMark = marks[v] + g.cost(v, adj)
            if (newMark < marks[adj]):
                heappush(temp, Node(adj, newMark))
                marks[adj] = newMark
                prev[adj] = v

        cst = heappop(temp)
        newv = cst.idx
        g.constant[newv] = 1
        v = newv

    path = [v]
    while (v != start):
        path.append(prev[v])
        v = prev[v]
    path = list(reversed(path))
    g.marks = marks

    return marks[final], path

     

        
