import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def dataPreparation(path):
    # Load the data
    data = pd.read_csv(path)

    # calculate the percentage change in "flow1"
    data['flow1_pct_change'] = data.groupby('ccode1')['flow1'].pct_change()

    # take only the rows for the year 2014
    data_2014 = data[data['year'] == 2014]

    # remove the rows where the percentage change is NaN
    data_2014 = data_2014.dropna(subset=['flow1_pct_change'])

    # remove useless variables
    data_2014 = data_2014.drop(
        columns=['year', 'flow1', 'flow2', 'smoothflow1', 'smoothflow2', 'smoothtotrade', 'spike1', 'spike2', 'dip1',
                 'dip2', 'trdspike', 'tradedip', 'bel_lux_alt_flow1', 'bel_lux_alt_flow2', 'china_alt_flow1',
                 'china_alt_flow2', 'source1', 'source2', 'version'])

    # remove rows having "flow1_pct_change" NaN or infinite
    data_2014 = data_2014[~data_2014['flow1_pct_change'].isnull()]
    data_2014 = data_2014[~data_2014['flow1_pct_change'].isin([np.inf, -np.inf])]

    # remove duplicate rows
    data_2014 = data_2014.drop_duplicates()

    return data_2014

def drawGraph(graph, out_path, ccode_dict, color_map=None):
    # Get positions for the nodes in the graph
    pos = nx.spring_layout(graph)

    # Draw the graph in 2 separate subplots, one for positive weights and one for negative weights
    plt.figure(figsize=(20, 10))
    plt.title("Graph Representation of DataFrame")

    # edge color and edge width should be based on the weight of the edge. The higher the weight, the darker the edge color and the thicker the edge. Except for the edges with negative weights, they should be colored red.

    # Separate the edges based on weight
    positive_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['weight'] > 0]
    negative_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['weight'] <= 0]

    # Get weights for edges
    positive_weights = [graph[u][v]['weight'] for u, v in positive_edges]
    negative_weights = [graph[u][v]['weight'] for u, v in negative_edges]

    # set the maximum and minimum weight to +1 and -1
    positive_weights = [min(weight, 1) for weight in positive_weights]
    negative_weights = [max(weight, -1) for weight in negative_weights]

    # First subplot
    plt.subplot(121, title="Positive Edges")

    # Draw positive edges
    nx.draw_networkx_edges(
        graph, pos,
        edgelist=positive_edges,
        width=[max(0.1, abs(weight) * 3) for weight in positive_weights],  # Adjust the scaling factor for weights
        edge_color=positive_weights,
        edge_cmap=plt.cm.Blues,
        edge_vmin=0, edge_vmax=max(positive_weights)
    )
    # Draw the nodes with color mapping if color_map is not None
    if color_map is not None:
        # If color_map is a dictionary, extract colors in the order of nodes
        if isinstance(color_map, dict):
            node_colors = [color_map.get(node, 'black') for node in graph.nodes()]
        else:
            # If color_map is a list or a sequence, use it directly
            node_colors = color_map
    else:
        # Default color is black if color_map is None
        node_colors = 'black'

    # Use node_colors for the node_color parameter
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=100)
    # draw node labels on the graph. And move the labels a little bit away from the nodes
    nx.draw_networkx_labels(graph, pos, font_size=10, labels=ccode_dict, font_color='black', verticalalignment='bottom')

    # second subplot
    plt.subplot(122, title="Negative Edges")

    # Draw negative edges
    nx.draw_networkx_edges(
        graph, pos,
        edgelist=negative_edges,
        width=[max(0.1, abs(weight) * 3) for weight in negative_weights],  # Adjust the scaling factor for weights
        edge_color=negative_weights,
        edge_cmap=plt.cm.Reds,
        edge_vmin=min(negative_weights), edge_vmax=0
    )
    # Draw the nodes with color mapping if color_map is not None
    if color_map is not None:
        # If color_map is a dictionary, extract colors in the order of nodes
        if isinstance(color_map, dict):
            node_colors = [color_map.get(node, 'black') for node in graph.nodes()]
        else:
            # If color_map is a list or a sequence, use it directly
            node_colors = color_map
    else:
        # Default color is black if color_map is None
        node_colors = 'black'

    # Use node_colors for the node_color parameter
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=100)

    # draw node labels on the graph. And move the labels a little bit away from the nodes
    nx.draw_networkx_labels(graph, pos, font_size=10, labels=ccode_dict, font_color='black', verticalalignment='bottom')

    # Show the plot
    plt.savefig(out_path)
    plt.close()

def drawSinglePlot(graph, out_path, ccode_dict, color_map=None):
    # Get positions for the nodes in the graph
    pos = nx.spring_layout(graph)

    # Draw the graph in a single plot. Using a palette to represent the weights of the edges. And use the color_map to color the nodes.
    plt.figure(figsize=(20, 10))
    plt.title("Graph Representation of DataFrame")

    # Draw the edges with a color palette based on the weight of the edge
    edges = graph.edges()
    weights = [graph[u][v]['weight'] for u, v in edges]
    edge_colors = [weight for weight in weights]

    # Draw the edges
    nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color=edge_colors, edge_cmap='magma', edge_vmin=min(weights), edge_vmax=max(weights))

    # Draw the nodes with color mapping if color_map is not None
    if color_map is not None:
        # If color_map is a dictionary, extract colors in the order of nodes
        if isinstance(color_map, dict):
            node_colors = [color_map.get(node, 'black') for node in graph.nodes()]
        else:
            # If color_map is a list or a sequence, use it directly
            node_colors = color_map
    else:
        # Default color is black if color_map is None
        node_colors = 'black'

    # Use node_colors for the node_color parameter
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=100)

    # draw node labels on the graph. And move the labels a little bit away from the nodes
    nx.draw_networkx_labels(graph, pos, font_size=10, labels=ccode_dict, font_color='black', verticalalignment='bottom')

    # Show the plot
    plt.savefig(out_path)
    plt.close()
