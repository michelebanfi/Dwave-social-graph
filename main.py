import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from dwave.system import LeapHybridCQMSampler
from dimod import ConstrainedQuadraticModel, Binary
from sklearn.preprocessing import MinMaxScaler
import geopandas as gpd

from Utils import dataPreparation, drawGraph, drawSinglePlot

data = dataPreparation("Dyadic_COW_4.0.csv")

# scale the flow1_pct_change variable
scaler = MinMaxScaler()
data['flow1_pct_change_normalized'] = scaler.fit_transform(data[['flow1_pct_change']])

# Create an empty directed graph
G = nx.Graph()

# Add edges to the graph
for idx, row in data.iterrows():
    G.add_edge(row["ccode1"], row["ccode2"], weight=row["flow1_pct_change"])

# create a dictionary between country codes and country names
ccodes = data['ccode1'].unique()
cnames = data['importer1'].unique()

ccode_dict = dict(zip(ccodes, cnames))
ccode_dictReversed = {v: k for k, v in ccode_dict.items()}

# Draw the graph
print("Plotting the graph")
drawGraph(G, "Media/graph.png", ccode_dict)

# print the number of nodes
print("Nodes present in the graph: ", len(G.nodes))

# create a method to remove a node from the graph and his edges
def remove_node(G, node):
    # remove the node from the graph
    G.remove_node(node)

# # list of removable nodes
# list = ["San Marino", "East Timor", "Monaco", "Yugoslavia", "Andorra", "Cape Verde", "Marshall Islands", "Vietnam",
#         "Sao Tome and Principe", "Kiribati", "Tuvalu", "Nauru", "Palau", "Solomon Islands", "Vanuatu", "Comoros",
#         "Seychelles", "Mauritius", "Maldives", "Liechtenstein", "Federated States of Micronesia", "Vietnam",
#         "Liechtenstein", "South Sudan", "Kazakhstan", "Venezuela", "Nicaragua"]

# list of foundamental nodes
f_list = ["United States of America", "Italy", "Germany", "Canada", "China", "Russia", "Japan", "Brazil", "India",
          "France", "United Kingdom", "Australia", "Spain", "Mexico", "South Korea", "Netherlands", "Turkey",
          "Saudi Arabia", "Indonesia", "Switzerland", "Sweden", "Norway", "Denmark", "Finland", "Belgium",
          "Austria", "Poland", "Czech Republic", "Portugal", "Greece", "Ireland", "Hungary", "Romania", "Bulgaria",
          "Croatia", "Slovenia", "Slovakia", "Estonia", "Latvia", "Lithuania", "Cyprus", "Malta", "Luxembourg",
          "Iceland", "New Zealand", "Singapore", "Taiwan", "South Africa", "Argentina", "Chile", "Colombia", "Peru",
          "Thailand", "Philippines", "Malaysia"]

# keep only the nodes that are in the list and remove the others
nodesToKeep = []
nodesToRemove = []
for node in f_list:
    node = ccode_dictReversed[node]
    nodesToKeep.append(node)

# iterate over the nodes and remove the ones that are not in the list
for node in G.nodes:
    if node not in nodesToKeep:
        nodesToRemove.append(node)

# remove nodes
for node in nodesToRemove:
    try:
        remove_node(G, node)
    except:
        pass

# print the number of nodes
print("Remaining nodes: ", len(G.nodes))
print("Remaining edges: ", len(G.edges))

# recalculate ccode_dict, accounting for the removed nodes
new_ccode_dict = {k: v for k, v in ccode_dict.items() if k in G.nodes}

# Draw the graph
print("Plotting the graph")
drawGraph(G, "Media/prunedGraph.png", new_ccode_dict)

# Define the number of clusters
num_clusters = 3  # You can change this to 4 or another number

# Create binary variables for each node and each cluster
cqm = ConstrainedQuadraticModel()
node_cluster_vars = {node: [Binary(f'x_{node}_{cluster}') for cluster in range(num_clusters)] for node in G.nodes}

# Add constraints to ensure each node is assigned to exactly one cluster
for node, vars in node_cluster_vars.items():
    cqm.add_constraint(sum(vars) == 1, label=f'one_cluster_{node}')

# Add the objective function to maximize intra-cluster similarity and minimize inter-cluster similarity
objective = 0
for u, v, data in G.edges(data=True):
    weight = data['weight']
    for i in range(num_clusters):
        # Nodes in the same cluster
        objective += weight * node_cluster_vars[u][i] * node_cluster_vars[v][i]
        # Nodes in different clusters
        for j in range(i+1, num_clusters):
            objective -= weight * node_cluster_vars[u][i] * node_cluster_vars[v][j]
            objective -= weight * node_cluster_vars[u][j] * node_cluster_vars[v][i]

cqm.set_objective(objective)

# Solve the problem
sampler = LeapHybridCQMSampler(token="DEV-c656c45b76fe02df536f0ce348eb602bcab9f1de")
sampleset = sampler.sample_cqm(cqm)

# Get the best solution
best_solution = sampleset.first.sample

# Print the clusters
clusters = {i: [] for i in range(num_clusters)}
for node in G.nodes:
    for i in range(num_clusters):
        if best_solution[f'x_{node}_{i}']:
            clusters[i].append(node)

print("Clusters: ", clusters)

# Convert clusters to country names
clusters_named = {i: [ccode_dict[node] for node in cluster] for i, cluster in clusters.items()}
print("Named Clusters: ", clusters_named)

# check cluster length
for i in range(num_clusters):
    print(len(clusters_named[i]))

# Ensure each nation is in only one cluster
seen_nodes = set()
for i in range(num_clusters):
    clusters[i] = [node for node in clusters[i] if node not in seen_nodes]
    seen_nodes.update(clusters[i])

# Check if any nodes are missing from all clusters
all_nodes = set(G.nodes)
missing_nodes = all_nodes - seen_nodes
if missing_nodes:
    print(f"Warning: These nodes were not assigned to any cluster: {missing_nodes}")
    # Assign missing nodes to the smallest cluster
    smallest_cluster = min(clusters, key=lambda k: len(clusters[k]))
    clusters[smallest_cluster].extend(missing_nodes)

print("Clusters after ensuring uniqueness: ", clusters)

# check cluster length
tot = 0
for i in range(num_clusters):
    print(len(clusters[i]))
    tot += len(clusters[i])
print("Total number of elements: ",tot)

# save the clustering solution
with open("clusters.txt", "w") as f:
    for i in range(num_clusters):
        f.write(f"Cluster {i}:\n")
        for node in clusters[i]:
            f.write(f"{ccode_dict[node]}\n")
        f.write("\n")

# save the clustering solution
with open("Media/clusters.txt", "w") as f:
    for i in range(num_clusters):
        f.write(f"Cluster {i}:\n")
        for node in clusters[i]:
            f.write(f"{ccode_dict[node]}\n")
        f.write("\n")

# Draw the graph with the clusters
# we have the best solution, now we need to plot the graph with the best solution. Use a different color for each cluster
color_map = []
for node in G.nodes:
    if node in clusters[0]:
        color_map.append('blue')
    elif node in clusters[1]:
        color_map.append('orange')
    elif node in clusters[2]:
        color_map.append('green')
    else:
        color_map.append('black')

# Draw the graph
print("Plotting the graph")
drawSinglePlot(G, "Media/clustered_prunedGraph.png", new_ccode_dict, color_map)

print("Plotting the world map")

# Convert clusters to country names
clusters_named = {i: [ccode_dict[node] for node in cluster] for i, cluster in clusters.items()}
print("Named Clusters: ", clusters_named)

# Define a dictionary that maps each cluster to a specific color
cluster_colors = {
    0: 'blue',
    1: 'green',
    2: 'orange'
}

# Load the world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Create a new column with the cluster number
world['cluster'] = world['name'].map({country: cluster for cluster, countries in clusters_named.items() for country in countries})

# Assign colors using the cluster_colors dictionary. If the state is not present, assign 'lightgrey'
world['color'] = world['cluster'].map(cluster_colors).fillna('lightgrey')

plt.figure(figsize=(20, 15))
# Plot the world map
world.plot(column='cluster', color=world['color'], figsize=(15, 10), missing_kwds={"color": "lightgrey"})
# remove legend and axis
plt.axis('off')
# remove legend
plt.legend().set_visible(False)
plt.savefig('Media/world_map.png')
plt.close()