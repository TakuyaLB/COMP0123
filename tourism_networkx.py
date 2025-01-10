import networkx as nx
import os
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities
from community import community_louvain
from collections import defaultdict
import seaborn as sns
import numpy as np
from netgraph import Graph, InteractiveGraph, EditableGraph

import warnings
warnings.filterwarnings("ignore")

def load_graph(csv_path):
    return nx.read_weighted_edgelist(csv_path, delimiter=",")

def visualize_graph(graph):
    nx.draw(graph, with_labels=True, font_weight='bold', width=0.01)
    plt.savefig(f"appendices/standard_graph_vis/standard graph visualization.png")
    plt.close()
    
    # Calculate betweenness centrality
    betweenness = nx.betweenness_centrality(graph)

    # Sort nodes by betweenness centrality in descending order
    sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)

    # Print the top 10 nodes with the highest betweenness centrality
    print("Top 10 nodes with highest betweenness centrality:")
    for i, (node, centrality) in enumerate(sorted_nodes[:10], start=1):
        print(f"{i}: Node {node}, Betweenness Centrality: {centrality:.4f}")

def degree_distribution(graph):
    # Compute degree distribution
    degrees = [degree for node, degree in graph.degree()]  # List of degrees

    # Plot degree distribution
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), align='left', rwidth=0.8)
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.savefig(f"appendices/standard_graph_vis/standard graph degree distribution.png")
    plt.close()

def degree_assortativity(graph):
    # Degree assortativity coefficient
    assortativity = nx.degree_assortativity_coefficient(graph)
    print("Degree Assortativity Coefficient:", assortativity)
    
    # Compute degree mixing matrix
    mixing_matrix = nx.degree_mixing_matrix(graph)

    # Visualize the matrix
    plt.imshow(mixing_matrix, cmap='Blues', interpolation='none')
    plt.colorbar(label="Edge Fraction")
    plt.title("Degree Mixing Matrix")
    plt.xlabel("Degree")
    plt.ylabel("Degree")
    plt.savefig(f"appendices/standard_graph_vis/standard graph degree assortativity matrix.png")
    plt.close()
    
def visualise_communities_greedy(communities, graph, graph_version):
    colors = []
    for node in graph.nodes():
        for i, community in enumerate(communities):
            if node in community:
                colors.append(i)
                break

    # Draw graph with community coloring
    nx.draw(graph, node_color=colors, cmap=plt.cm.tab10, with_labels=True)
    plt.title(f"{graph_version} Greedy Community Detection")
    plt.savefig(f"appendices/greedy_community_detection/{graph_version} greedy communities.png")
    plt.close()

def visualise_communities_louvain(partition, graph, graph_version):
    community_to_color = {
    0 : 'tab:blue',
    1 : 'tab:orange',
    2 : 'tab:green',
    3 : 'tab:red',
    4 : 'tab:purple',
    5 : 'tab:brown',
    6 : 'tab:pink',
}

    node_color = {node: community_to_color[community_id] for node, community_id in partition.items()}
    
    Graph(graph, node_color=node_color, node_edge_width=0, edge_alpha=0.1, 
          node_layout="community", node_layout_kwargs=dict(node_to_community=partition), 
          edge_cmap='viridis', edge_layout_kwargs=dict(k=2000), 
    ) 
    plt.title(f"{graph_version} Louvain Community Detection")
    plt.savefig(f"appendices/louvain_community_detection/{graph_version} louvain communities.png")
    plt.close()

def compute_communities_greedy(graph, graph_version):
    communities = list(greedy_modularity_communities(graph))
    visualise_communities_greedy(communities, graph, graph_version)
    return communities

def compute_communities_louvain(graph, graph_version):
    partition = community_louvain.best_partition(graph, random_state=42)
    visualise_communities_louvain(partition, graph, graph_version)
    modularity = community_louvain.modularity(partition, graph)
    communities = defaultdict(list)
    for node, community in partition.items():
        communities[community].append(node)
    return list(communities.values()), modularity

def analyze_communities(graph, communities):
    size_density = []
    centrality_data = {
        'betweenness': {},
        'degreeness': {},
        'eigenvector': {}
    }

    for i, community in enumerate(communities):
        subgraph = graph.subgraph(community)
        size = len(community)
        density = nx.density(subgraph)
        size_density.append((size, density))

        centralities = {
            'betweenness': nx.betweenness_centrality(subgraph),
            'degreeness': nx.degree_centrality(subgraph),
            'eigenvector': nx.eigenvector_centrality(subgraph)
        }

        for key, values in centralities.items():
            centrality_data[key][i] = sorted(values.items(), key=lambda x: x[1], reverse=True)

    return size_density, centrality_data

def plot_top_centrality(centrality_ranking, metric_name, community_index, graph_version, community_method):
    top_nodes = [node for node, _ in centrality_ranking[:10]]
    top_values = [value for _, value in centrality_ranking[:10]]
    if graph_version == "Standard graph" and community_method == "louvain" and metric_name == "betweenness": 
        print(f"{metric_name} centrality")
        print(f"Community {community_index}:")
        for i in range(len(top_nodes)):
            print(f"{i}. {top_nodes[i]} (Centrality: {top_values[i]})")

    plt.figure(figsize=(10, 6))
    plt.bar(top_nodes, top_values, color='skyblue')
    plt.title(f"Top 10 Nodes by {metric_name.capitalize()} Centrality \n(Community {community_index}, {graph_version})")
    plt.xlabel("Node")
    plt.ylabel(f"{metric_name.capitalize()} Centrality")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"appendices/intra_community_centrality/top {metric_name} community {community_index} {graph_version} {community_method}.png")
    plt.close()

def display_community_metrics(size_density, centrality_data, modularity=None, graph_version="Standard graph", community_method="greedy"):
    print(f"{len(size_density)} communities")
    for i, (size, density) in enumerate(size_density):
        print(f"Community {i}: size {size}, density {density:.4f}")
    
    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(size_density)), [size for (size, _) in size_density], color="skyblue")
    plt.title(f"{graph_version} Community Size Distribution ({community_method})")
    plt.xlabel("Community ID")
    plt.ylabel("Size")
    plt.savefig(f"appendices/community_size_distribution/{graph_version} {community_method} community size distribution.png")
    plt.close()

    if modularity is not None:
        print(f"Modularity: {modularity:.4f}")

    for metric, communities in centrality_data.items():
        for i, ranking in communities.items():
            plot_top_centrality(ranking, metric, i, graph_version, community_method)

def remove_top_nodes(graph, centralities, k):
    nodes_to_remove = [node for node, _ in centralities[:k]]
    modified_graph = graph.copy()
    modified_graph.remove_nodes_from(nodes_to_remove)
    
    # Visualize the network before and after node removal
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    nx.draw(graph, ax=axes[0], with_labels=True)
    nx.draw(modified_graph, ax=axes[1], with_labels=True)
    axes[0].set_title("Before Node Removal")
    axes[1].set_title("After Node Removal")
    plt.title("Node Removal")
    plt.savefig(f"appendices/removal_comparisons/node removal comparison.png")
    plt.close()
    
    return modified_graph

def remove_top_edges(graph, edge_betweenness, k):
    edges_to_remove = [edge for edge, _ in edge_betweenness[:k]]
    modified_graph = graph.copy()
    modified_graph.remove_edges_from(edges_to_remove)
    
    # Visualize the network before and after edge removal
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    nx.draw(graph, ax=axes[0], with_labels=True)
    nx.draw(modified_graph, ax=axes[1], with_labels=True)
    axes[0].set_title("Before Edge Removal")
    axes[1].set_title("After Edge Removal")
    plt.title("Edge Removal")
    plt.savefig(f"appendices/removal_comparisons/edge removal comparison.png")
    plt.close()
    
    return modified_graph

def compute_graph_metrics(graph):
    metrics = {
        'largest_connected_component': len(max(nx.connected_components(graph), key=len)),
        'average_shortest_path_length': nx.average_shortest_path_length(graph),
        'diameter': nx.diameter(graph)
    }
    return metrics

def print_graph_metrics(metrics):
    for key, value in metrics.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")
        
def centrality_distributions(graph, graph_version):
    centralities = {
            'Betweenness': nx.betweenness_centrality(graph),
            'Degree': nx.degree_centrality(graph),
            'Eigenvector': nx.eigenvector_centrality(graph)
        }
    for key, value in centralities.items():
        np_values = np.fromiter(value.values(), dtype=float) 
        sns.histplot(np_values, kde=True)
        sns.set_theme(rc={"figure.figsize":(10, 6)})
        plt.title(f"{graph_version} {key} Centrality Distribution")
        plt.xlabel(f"{key} Centrality")
        plt.ylabel("Frequency")
        plt.savefig(f"appendices/centrality_distributions/{graph_version} {key} Centrality Distribution.png")
        plt.close()
        
def centrality_highlight(graph, graph_version):
    centralities = {
            'Betweenness': nx.betweenness_centrality(graph),
            'Degree': nx.degree_centrality(graph),
            'Eigenvector': nx.eigenvector_centrality(graph)
        }
    for key, value in centralities.items():
        nx.draw(graph, node_size=[v * 1000 for v in value.values()], with_labels=True)
        plt.title(f"{graph_version} Node Size Based on {key} Centrality")
        plt.savefig(f"appendices/centrality_highlighted_graphs/{graph_version} {key} Centrality Graph.png")
        plt.close()

def analyze_graph(graph, description):
    print(f"\n{description} analysis")

    # Greedy community detection
    greedy_communities = compute_communities_greedy(graph, description)
    greedy_size_density, greedy_centrality_data = analyze_communities(graph, greedy_communities)
    print("Greedy Community Detection:")
    display_community_metrics(greedy_size_density, greedy_centrality_data, graph_version=description, community_method="greedy")

    # Louvain community detection
    louvain_communities, modularity = compute_communities_louvain(graph, description)
    louvain_size_density, louvain_centrality_data = analyze_communities(graph, louvain_communities)
    print("Louvain Community Detection:")
    display_community_metrics(louvain_size_density, louvain_centrality_data, modularity, graph_version=description, community_method="louvain")

    # Graph metrics
    print("Graph Metrics:")
    print_graph_metrics(compute_graph_metrics(graph))
    
    # Centrality distributions
    centrality_distributions(graph, description)
    
    # Node importance
    centrality_highlight(graph, description)

def main():
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    csv_path = os.path.join(__location__, "tourism_network_2021_networkx.csv")

    G = load_graph(csv_path)
    visualize_graph(G)
    degree_distribution(G)
    degree_assortativity(G)

    # Standard graph analysis
    analyze_graph(G, "Standard graph")

    # Compute centralities for node and edge removal
    graph_betweenness = sorted(nx.betweenness_centrality(G).items(), key=lambda x: x[1], reverse=True)
    graph_edge_betweenness = sorted(nx.edge_betweenness_centrality(G).items(), key=lambda x: x[1], reverse=True)

    # Node removal analysis
    G_node_removed = remove_top_nodes(G, graph_betweenness, 5)
    analyze_graph(G_node_removed, "Node removal (top 5 betweenness nodes)")

    # Edge removal analysis
    G_edge_removed = remove_top_edges(G, graph_edge_betweenness, 5)
    analyze_graph(G_edge_removed, "Edge removal (top 5 betweenness edges)")

if __name__ == "__main__":
    main()

# note for community 0 from louvain community detection there are only 7 members, all with the same degreeness and eigenvector centrality values and all with betweenness centrality 0 so the graphs are weird