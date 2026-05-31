import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

students = ["Amit", "Riya", "Neha", "John", "Sara", "Ali", "Priya", "Karan", "Meena", "Ravi"]
G.add_nodes_from(students)

friendships = [
    ("Amit", "Riya"),
    ("Amit", "Neha"),
    ("Amit", "John"),
    ("Riya", "Neha"),
    ("Riya", "Sara"),
    ("Riya", "John"),
    ("Neha", "Sara"),
    ("Sara", "Ali"),
    ("John", "Ali"),
    ("Priya", "Karan"),
    ("Karan", "Meena"),
    ("Meena", "Ravi"),
    ("Ravi", "Priya"),
    ("Amit", "Priya"),
    ("Neha", "Meena")
]

G.add_edges_from(friendships)

plt.figure(figsize=(7,5))
nx.draw(G, with_labels=True, node_size=2000)
plt.title("Student Friendship Network")
plt.show()

print("The nodes are the students: ")
for i in range(len(students)):
  print(students[i])
print("The edges are the friendships: ")
for j in range(len(friendships)):
  print(friendships[j])
name_to_idx = {name: i for i, name in enumerate(students)}
num_nodes = len(students)
adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)


for u, v in friendships:
    i, j = name_to_idx[u], name_to_idx[v]
    adj_matrix[i, j] = 1
    adj_matrix[j, i] = 1  

print(adj_matrix)
#We observe the matrix and the graph show the same network


#####DEGREE######
# Manually 
graph = {
    "Amit":  ["Riya", "Neha", "John", "Priya"],
    "Riya":  ["Amit", "Neha", "Sara", "John"],
    "Neha":  ["Amit", "Riya", "Sara", "Meena"],
    "John":  ["Amit", "Riya", "Ali"],
    "Sara":  ["Riya", "Neha", "Ali"],
    "Ali":   ["Sara", "John"],
    "Priya": ["Karan", "Ravi", "Amit"],
    "Karan": ["Priya", "Meena"],
    "Meena": ["Karan", "Ravi", "Neha"],
    "Ravi":  ["Meena", "Priya"]
}

def get_degree(graph_data, node):
    if node not in graph_data:
        return 0
    return len(graph_data[node])

def get_neighbours(graph_data, node):
    return graph_data.get(node, [])

def is_connected_bfs(graph_data, start_node, target_node):
    if start_node == target_node:
        return True

    if start_node not in graph_data or target_node not in graph_data:
        return False

    queue = [start_node]
    visited = {start_node}

    while queue:
        current_node = queue.pop(0)
        if current_node == target_node:
            return True

        for neighbour in graph_data[current_node]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

    return False

# Built in Functions
degree = dict(G.degree())
print("Degree of each student:")
for student, deg in degree.items():
    print(student," : ", {deg})

max_friends = max(degree.values())
min_friends = min(degree.values())

print("Max Friends", max_friends)
print("Min Friends", min_friends)

density = nx.density(G)

# Connectivity check
is_connected = nx.is_connected(G)
print(
    f"Is the network connected? {'Yes' if is_connected else 'No, it is disconnected.'}\n"
)

# Task 5: Identify Connected Components

print("--- Task 5: Connected Components ---")
components = list(nx.connected_components(G))
for i, comp in enumerate(components, 1):
    print(f"Component {i}: {', '.join(comp)}")
print()

# Task 6: Improve Visualization

plt.figure(figsize=(10, 7))

# 1. Node size depends on the number of friends (degree * 300 for scaling)
node_sizes = [degree[node] * 300 for node in G.nodes()]

# 2. Node color depends on the connected component
# Assign a unique color to each component dynamically
component_colors = ["lightcoral", "lightgreen", "gold", "violet"]
node_color_map = []

for node in G.nodes():
    for idx, component in enumerate(components):
        if node in component:
            node_color_map.append(component_colors[idx % len(component_colors)])
            break
pos = nx.spring_layout(G, seed=42)
# Draw the advanced network
nx.draw_networkx_nodes(
    G, pos, node_color=node_color_map, node_size=node_sizes, alpha=0.9
)
nx.draw_networkx_edges(G, pos, edge_color="dimgray", width=1.5)
nx.draw_networkx_labels(
    G, pos, font_size=10, font_weight="bold", font_family="sans-serif"
)

plt.title("Improved Student Friendship Network", fontsize=14, fontweight="bold")
plt.axis("off")
plt.show()

