"""
Visualization utilities for Mars Rover pathfinding solutions.
"""
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


class NetworkVisualizer:
    """Utility class for visualizing the Mars network and pathfinding solutions."""
    
    SCENARIO_CONFIGS = {
        'scientific': {
            'title': 'Scientific Sample Collection - Radiation-Aware Pathfinding',
            'path_color': 'lime',
            'description': (
                'Objective: Reach mineral site with minimal radiation\n'
                'Constraints:\n'
                '- Max radiation per node: 0.6\n'
                '- Minimize total radiation exposure\n'
                '- Protect scientific instruments'
            ),
            'legend_items': [
                ('lightblue', 'Low Radiation (< 0.3)'),
                ('orange', 'Medium Radiation (0.3 - 0.6)'),
                ('red', 'High Radiation (> 0.6)')
            ]
        },
        'solar': {
            'title': 'Solar Power Route - Energy-Efficient Pathfinding',
            'path_color': 'yellow',
            'description': (
                'Objective: Reach charging station via sunny areas\n'
                'Constraints:\n'
                '- Min solar coverage: 0.5\n'
                '- Maximize battery recharge\n'
                '- Avoid shadowed regions'
            ),
            'legend_items': [
                ('lightblue', 'High Solar Coverage (> 0.7)'),
                ('orange', 'Medium Solar Coverage (0.4 - 0.7)'),
                ('red', 'Low Solar Coverage (< 0.4)')
            ]
        },
        'communication': {
            'title': 'Communication Relay - Visibility-Optimized Pathfinding',
            'path_color': 'cyan',
            'description': (
                'Objective: Maintain orbiter communication\n'
                'Constraints:\n'
                '- Max 2 consecutive blind hops\n'
                '- Prefer visible nodes\n'
                '- Enable data transmission'
            ),
            'legend_items': [
                ('lightblue', 'Has Orbiter Visibility'),
                ('red', 'No Visibility (Blind Spot)')
            ]
        }
    }
    
    TERRAIN_MARKERS = {
        'crater': 's',    # Square
        'plain': 'o',     # Circle
        'canyon': '^',    # Triangle
        'plateau': 'D'    # Diamond
    }
    
    def __init__(self, figsize: Tuple[int, int] = (16, 12)):
        """Initialize visualizer with figure size."""
        self.figsize = figsize
        plt.close('all')
    
    def display_initial_network(self, network) -> None:
        """
        Display the initial state of the Mars network.
        
        Args:
            network: The Mars network instance
        """
        plt.close('all')
        fig = plt.figure(figsize=self.figsize)
        
        # Get positions and basic visualization data
        pos = network.get_node_positions()
        
        # Draw the basic network structure
        node_colors = []
        for node_id in sorted(network.nodes.keys()):
            station = network.nodes[node_id]
            # Color by terrain type
            terrain_colors = {
                'crater': 'red',
                'plain': 'lightgreen',
                'canyon': 'orange',
                'plateau': 'lightblue'
            }
            node_colors.append(terrain_colors.get(station.terrain_type, 'gray'))
        
        nx.draw_networkx_nodes(network.graph, pos, 
                             node_color=node_colors,
                             node_size=1200,
                             edgecolors='black',
                             linewidths=2)
        nx.draw_networkx_edges(network.graph, pos, edge_color='gray', width=2)
        
        # Draw node labels with station info
        labels = {}
        for node_id, station in network.nodes.items():
            labels[node_id] = f"S{node_id}\n{station.terrain_type[:4]}"
        
        nx.draw_networkx_labels(network.graph, pos, labels, font_size=8, font_weight='bold')
        
        # Draw edge weights
        edge_labels = {}
        for (u, v, data) in network.graph.edges(data=True):
            edge_labels[(u, v)] = f"{data['weight']:.0f}"
        nx.draw_networkx_edge_labels(network.graph, pos, 
                                   edge_labels=edge_labels,
                                   font_size=7)
        
        # Add legend for terrain types
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='red', markersize=12,
                      markeredgecolor='black', markeredgewidth=1,
                      label='Crater (difficult)'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='orange', markersize=12,
                      markeredgecolor='black', markeredgewidth=1,
                      label='Canyon (moderate)'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='lightblue', markersize=12,
                      markeredgecolor='black', markeredgewidth=1,
                      label='Plateau (safe)'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='lightgreen', markersize=12,
                      markeredgecolor='black', markeredgewidth=1,
                      label='Plain (easy)')
        ]
        
        plt.legend(handles=legend_elements, 
                  loc='center left',
                  bbox_to_anchor=(1, 0.5),
                  title='Terrain Types',
                  title_fontsize=12,
                  fontsize=10)
        
        # Add title and information
        plt.suptitle('Mars Rover Exploration Network - Initial State', 
                    fontsize=16, y=0.98, fontweight='bold')
        plt.figtext(0.02, 0.02, 
                   'Node Format: Station ID + Terrain Type\n'
                   'Edge weights represent traversal cost',
                   fontsize=10, style='italic',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close('all')
    
    def plot_scenario(self,
                     network,
                     scenario_type: str,
                     path: List[int],
                     start: int = None,
                     goal: int = None) -> None:
        """
        Plot a specific scenario with pathfinding solution.
        
        Args:
            network: The Mars network instance
            scenario_type: Type of scenario
            path: List of node IDs representing the found path
            start: Start node ID
            goal: Goal node ID
        """
        if scenario_type not in self.SCENARIO_CONFIGS:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
            
        config = self.SCENARIO_CONFIGS[scenario_type]
        
        plt.close('all')
        fig = plt.figure(figsize=self.figsize)
        
        # Get positions and colors for current scenario
        pos = network.get_node_positions()
        node_colors = network.get_node_colors(scenario_type)
        
        # Draw all edges in light gray
        nx.draw_networkx_edges(network.graph, pos, 
                             edge_color='lightgray',
                             style='dashed',
                             width=1)
        
        # Highlight path edges
        if path and len(path) > 1:
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            nx.draw_networkx_edges(network.graph,
                                 pos,
                                 edgelist=path_edges,
                                 edge_color=config['path_color'],
                                 width=4,
                                 style='solid')
        
        # Draw nodes with scenario colors
        nx.draw_networkx_nodes(network.graph, pos, 
                             node_color=node_colors,
                             node_size=1000,
                             edgecolors='black',
                             linewidths=1.5)
        
        # Highlight start and goal nodes
        if start is not None:
            nx.draw_networkx_nodes(network.graph, pos,
                                 nodelist=[start],
                                 node_color='lime',
                                 node_size=1400,
                                 edgecolors='darkgreen',
                                 linewidths=3)
        
        if goal is not None:
            nx.draw_networkx_nodes(network.graph, pos,
                                 nodelist=[goal],
                                 node_color='magenta',
                                 node_size=1400,
                                 edgecolors='purple',
                                 linewidths=3)
        
        # Draw node labels based on scenario
        labels = {}
        for node_id, station in network.nodes.items():
            if scenario_type == 'scientific':
                labels[node_id] = f"S{node_id}\n{station.radiation_level:.2f}"
            elif scenario_type == 'solar':
                labels[node_id] = f"S{node_id}\n{station.solar_coverage:.2f}"
            else:  # communication
                vis = "✓" if station.has_orbiter_visibility else "✗"
                labels[node_id] = f"S{node_id}\n{vis}"
        
        nx.draw_networkx_labels(network.graph, pos, labels, font_size=8, font_weight='bold')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=color, markersize=12,
                      markeredgecolor='black', markeredgewidth=1,
                      label=label)
            for color, label in config['legend_items']
        ]
        
        # Add path and special node markers to legend
        legend_elements.extend([
            plt.Line2D([0], [0], color=config['path_color'],
                      linewidth=4, label='Found Path'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='lime', markersize=14,
                      markeredgecolor='darkgreen', markeredgewidth=2,
                      label='START Node'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='magenta', markersize=14,
                      markeredgecolor='purple', markeredgewidth=2,
                      label='GOAL Node'),
            plt.Line2D([0], [0], color='lightgray',
                      linewidth=1, linestyle='--',
                      label='Other Connections')
        ])
        
        plt.legend(handles=legend_elements, 
                  loc='center left',
                  bbox_to_anchor=(1, 0.5),
                  title='Legend',
                  title_fontsize=12,
                  fontsize=10)
        
        # Add title and description
        plt.suptitle(config['title'], fontsize=16, y=0.98, fontweight='bold')
        plt.figtext(0.02, 0.02, config['description'], 
                   fontsize=10, style='italic',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Add path info
        if path:
            path_str = ' → '.join([f'S{n}' for n in path])
            cost = network.calculate_path_cost(path)
            plt.figtext(0.02, 0.92, 
                       f'Path: {path_str}\nTotal Cost: {cost:.1f} | Nodes: {len(path)}',
                       fontsize=10, fontweight='bold',
                       bbox=dict(facecolor='white', alpha=0.9, edgecolor='green'))
        
        plt.axis('off')
        plt.tight_layout()
        
    def save_plot(self, filename: str) -> None:
        """Save current plot to file."""
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        
    def show_plot(self) -> None:
        """Display current plot."""
        plt.show()
        
    def plot_comparison(self, network, astar_path: List[int], bfs_path: List[int],
                       scenario_type: str, start: int, goal: int) -> None:
        """
        Plot comparison of A* and BFS solutions side by side.
        
        Args:
            network: The Mars network instance
            astar_path: Path found by A*
            bfs_path: Path found by BFS
            scenario_type: Type of scenario
            start: Start node
            goal: Goal node
        """
        plt.close('all')
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        pos = network.get_node_positions()
        config = self.SCENARIO_CONFIGS.get(scenario_type, self.SCENARIO_CONFIGS['scientific'])
        node_colors = network.get_node_colors(scenario_type)
        
        paths = [('A* Algorithm', astar_path), ('BFS Algorithm', bfs_path)]
        
        for ax, (title, path) in zip(axes, paths):
            # Draw edges
            nx.draw_networkx_edges(network.graph, pos,
                                 edge_color='lightgray',
                                 style='dashed',
                                 width=1,
                                 ax=ax)
            
            # Draw path
            if path and len(path) > 1:
                path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                nx.draw_networkx_edges(network.graph, pos,
                                     edgelist=path_edges,
                                     edge_color=config['path_color'],
                                     width=3,
                                     ax=ax)
            
            # Draw nodes
            nx.draw_networkx_nodes(network.graph, pos,
                                 node_color=node_colors,
                                 node_size=600,
                                 ax=ax)
            
            # Highlight start/goal
            nx.draw_networkx_nodes(network.graph, pos,
                                 nodelist=[start],
                                 node_color='lime',
                                 node_size=800,
                                 ax=ax)
            nx.draw_networkx_nodes(network.graph, pos,
                                 nodelist=[goal],
                                 node_color='magenta',
                                 node_size=800,
                                 ax=ax)
            
            # Labels
            labels = {n: f"S{n}" for n in network.nodes}
            nx.draw_networkx_labels(network.graph, pos, labels, 
                                   font_size=8, ax=ax)
            
            # Title with metrics
            if path:
                cost = network.calculate_path_cost(path)
                ax.set_title(f"{title}\nCost: {cost:.1f} | Nodes: {len(path)}", 
                           fontsize=14, fontweight='bold')
            else:
                ax.set_title(f"{title}\nNo path found", fontsize=14)
            
            ax.axis('off')
        
        plt.suptitle(f'Algorithm Comparison - {scenario_type.capitalize()} Scenario',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()