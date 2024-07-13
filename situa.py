import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import ConstrainedQuadraticModel, Binary
from dwave.system import LeapHybridCQMSampler

# Load the data
data = pd.read_csv('Dyadic_COW_4.0.csv')

# calculate the percentage change in "flow1"
data['flow1_pct_change'] = data.groupby('ccode1')['flow1'].pct_change()

# take only the rows for the year 2014
data_2014 = data[data['year'] == 2014]

# remove the rows where the percentage change is NaN
data_2014 = data_2014.dropna(subset=['flow1_pct_change'])

# remove useless variables
data_2014 = data_2014.drop(columns=['year', 'flow1', 'flow2', 'smoothflow1', 'smoothflow2', 'smoothtotrade', 'spike1', 'spike2', 'dip1', 'dip2', 'trdspike', 'tradedip', 'bel_lux_alt_flow1', 'bel_lux_alt_flow2', 'china_alt_flow1', 'china_alt_flow2', 'source1', 'source2', 'version' ])

# remove rows having "flow1_pct_change" NaN or infinite
data_2014 = data_2014[~data_2014['flow1_pct_change'].isnull()]
data_2014 = data_2014[~data_2014['flow1_pct_change'].isin([np.inf, -np.inf])]

# remove duplicate rows
data_2014 = data_2014.drop_duplicates()

# Create an empty directed graph
G = nx.Graph()

# Add edges to the graph
for idx, row in data_2014.iterrows():
    G.add_edge(row["ccode1"], row["ccode2"], weight=row["flow1_pct_change"])

# Draw the graph
# Get positions for the nodes in the graph
pos = nx.spring_layout(G)

# create a dictionary between country codes and country names
ccodes = data_2014['ccode1'].unique()
cnames = data_2014['importer1'].unique()

ccode_dict = dict(zip(ccodes, cnames))
ccode_dictReversed  = {v: k for k, v in ccode_dict.items()}

# print the number of nodes
print(len(G.nodes))

# Draw the graph in 2 separate subplots, one for positive weights and one for negative weights
plt.figure(figsize=(20, 10))
plt.title("Graph Representation of DataFrame")

# edge color and edge width should be based on the weight of the edge. The higher the weight, the darker the edge color and the thicker the edge. Except for the edges with negative weights, they should be colored red.

# Separate the edges based on weight
positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > 0]
negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] <= 0]

# Get weights for edges
positive_weights = [G[u][v]['weight'] for u, v in positive_edges]
negative_weights = [G[u][v]['weight'] for u, v in negative_edges]

# set the maximum and minimum weight to +1 and -1
positive_weights = [min(weight, 1) for weight in positive_weights]
negative_weights = [max(weight, -1) for weight in negative_weights]

# First subplot
plt.subplot(121, title="Positive Edges")

# Draw positive edges
nx.draw_networkx_edges(
    G, pos,
    edgelist=positive_edges,
    width=[max(0.1, abs(weight)*3) for weight in positive_weights],  # Adjust the scaling factor for weights
    edge_color=positive_weights,
    edge_cmap=plt.cm.Blues,
    edge_vmin=0, edge_vmax=max(positive_weights)
)
# Draw the nodes
nx.draw_networkx_nodes(G, pos, node_color='black', node_size=100)
# draw node labels on the graph. And move the labels a little bit away from the nodes
nx.draw_networkx_labels(G, pos, font_size=10, labels=ccode_dict, font_color='black', verticalalignment='bottom')

# second subplot
plt.subplot(122, title="Negative Edges")

# Draw negative edges
nx.draw_networkx_edges(
    G, pos,
    edgelist=negative_edges,
    width=[max(0.1, abs(weight)*3) for weight in negative_weights],  # Adjust the scaling factor for weights
    edge_color=negative_weights,
    edge_cmap=plt.cm.Reds,
    edge_vmin=min(negative_weights), edge_vmax=0
)
nx.draw_networkx_nodes(G, pos, node_color='black', node_size=100)

# draw node labels on the graph. And move the labels a little bit away from the nodes
nx.draw_networkx_labels(G, pos, font_size=10, labels=ccode_dict, font_color='black', verticalalignment='bottom')

# Show the plot
plt.savefig("Media/original_graph.png")
plt.close()

# create a method to remove a node from the graph and his edges
def remove_node(G, node):
    # remove the node from the graph
    G.remove_node(node)
    print(len(G.edges))
    # # remove the edges connected to the node
    # for u, v in list(G.edges()):
    #     if u == node or v == node:
    #         G.remove_edge(u, v)

# list of removable nodes
list = ["San Marino", "East Timor", "Monaco", "Yugoslavia", "Andorra", "Cape Verde", "Marshall Islands", "Vietnam",
        "Sao Tome and Principe", "Kiribati", "Tuvalu", "Nauru", "Palau", "Solomon Islands", "Vanuatu", "Comoros",
        "Seychelles", "Mauritius", "Maldives", "Liechtenstein", "Federated States of Micronesia", "Vietnam",
        "Liechtenstein", "South Sudan", "Kazakhstan", "Venezuela", "Nicaragua"]

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
print("nodes: ", len(G.nodes))
print("edges: ", len(G.edges))

# Define the number of clusters
num_clusters = 3  # Example: 3 clusters

# Create binary variables for each node and each cluster
cqm = ConstrainedQuadraticModel()
node_cluster_vars = {node: [Binary(f'x_{node}_{cluster}') for cluster in range(num_clusters)] for node in G.nodes}

# Add constraints to ensure each node is assigned to exactly one cluster
for node, vars in node_cluster_vars.items():
    cqm.add_constraint(sum(vars) == 1, label=f'one_cluster_{node}')

# Add the objective function to minimize frustration
for u, v, data in G.edges(data=True):
    weight = data['weight']
    for i in range(num_clusters):
        cqm.set_objective(cqm.objective + weight * node_cluster_vars[u][i] * node_cluster_vars[v][i])

# Solve the problem
sampler = LeapHybridCQMSampler()
response = sampler.sample_cqm(cqm)

# Get the best solution
best_solution = response.first.sample

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