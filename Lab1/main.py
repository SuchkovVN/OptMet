import graphOpt as gopt
import networkx as nx
import matplotlib.pyplot as plt


G = gopt.Graph([[-1, 1, -1, 1, -1, 1, -1], 
                [-1, -1, 2, 4, 2, 1, 6],
                [-1, -1, -1, 2, -1, -1, 5],
                [-1, -1, -1, -1, 2, 3, -1],
                [-1, -1, -1, -1, -1, 2, 2],
                [-1, -1, -1, -1, -1, -1, 6],
                [-1, -1, -1, -1, -1, -1, -1]])
print(G.adjLists)
cost, path = gopt.Dijkstra(G, 0, 6)
print(cost)
print(path)

vgraph = nx.DiGraph()
vgraph.add_nodes_from([i for i in range(G.NumVertices)])

i = 0
for adj in G.adjLists:
    for j in adj:
        vgraph.add_edge(i, j, weight=G.cost(i, j))
    i += 1



pos = nx.nx_pydot.graphviz_layout(vgraph, root=0, prog='dot')

state_pos_cst = {n: (x+10, y+12) for n, (x,y) in pos.items() if G.constant[n] == True}
state_pos_tmp = {n: (x+10, y+12) for n, (x,y) in pos.items() if G.constant[n] == False}

nx.draw_networkx(vgraph, pos, with_labels=True, font_weight='bold')
edge_labels = nx.get_edge_attributes(vgraph, 'weight')
path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
nx.draw_networkx_edges(vgraph, pos, edgelist=path_edges, style="solid", width=2.5)
nx.draw_networkx_edge_labels(vgraph, pos, edge_labels=edge_labels)
node_labels_cst = {n: G.marks[n] for n in vgraph.nodes if G.constant[n] == True}
node_labels_tmp = {n: G.marks[n] for n in vgraph.nodes if G.constant[n] == False}
nx.draw_networkx_labels(vgraph, state_pos_cst, labels = node_labels_cst, font_color='blue')
nx.draw_networkx_labels(vgraph, state_pos_tmp, labels = node_labels_tmp, font_color='red')
plt.show()