import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# https://correlatesofwar.org/data-sets/bilateral-trade/
# https://github.com/dwave-examples/structural-imbalance-notebook/blob/master/01-structural-imbalance-overview.ipynb

# Load the data
data = pd.read_csv('Dyadic_COW_4.0.csv')

# create a dictionary between country codes and country names
ccodes = data['ccode1'].unique()
cnames = data['importer1'].unique()

ccode_dict = dict(zip(ccodes, cnames))

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
pos = nx.spring_layout(G)  # You can try different layouts like circular_layout, shell_layout, etc.

weights = [G[u][v]['weight'] for u, v in G.edges]

# Draw nodes and edges with almost transparent color
plt.figure(figsize=(30, 30))

# edge color and edge width should be based on the weight of the edge. The higher the weight, the darker the edge color and the thicker the edge.
nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=200, font_size=1, font_weight='bold', edge_color=weights, width=[max(0.1, abs(weight)*10) for weight in weights], edge_cmap=plt.cm.Blues, edge_vmin=-1, edge_vmax=1)

# nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f'{d["weight"]}' for u, v, d in G.edges(data=True)})
# nx.draw_networkx_edges(G, pos, width=[max(1, abs(weight)/2) for weight in weights], edge_color='gray')

# draw node labels
nx.draw_networkx_labels(G, pos, font_size=5)

# Show the plot
plt.title("Graph Representation of DataFrame")
plt.savefig("graph.png")
plt.show()

