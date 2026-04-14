"""
Mars exploration network representation for pathfinding.
"""
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from src.models.node import MarsStation


class MarsNetwork:
    """Represents the Mars exploration station network."""
    
    def __init__(self):
        """Initialize empty network."""
        self.graph = nx.Graph()
        self.nodes: Dict[int, MarsStation] = {}
    
    def add_node(self, station: MarsStation) -> None:
        """
        Add a station to the network.
        
        Args:
            station: MarsStation instance to add
        """
        self.nodes[station.id] = station
        self.graph.add_node(station.id)
    
    def add_link(self, node1_id: int, node2_id: int) -> None:
        """
        Add a link between stations with calculated cost.
        
        Args:
            node1_id: ID of first station
            node2_id: ID of second station
        """
        if node1_id not in self.nodes or node2_id not in self.nodes:
            raise ValueError("Both stations must exist in the network")
            
        station1 = self.nodes[node1_id]
        station2 = self.nodes[node2_id]
        
        # Calculate link cost based on traversal difficulty
        cost = station1.get_traversal_cost(station2)
        
        # Add edge with cost as weight
        self.graph.add_edge(node1_id, node2_id, weight=cost)
    
    @classmethod
    def create_mars_network(cls) -> 'MarsNetwork':
        """
        Create the Mars exploration network with hexagonal topology.
        
        Returns:
            MarsNetwork: A new network with predefined Mars stations
        """
        network = cls()
        
        # Define station data: (id, x, y, terrain, radiation, solar, visibility)
        stations_data = [
            # Central Hub Area
            (0, 300, 300, 'plateau', 0.20, 0.90, True),   # Base Camp
            (1, 400, 300, 'plateau', 0.30, 0.85, True),   # Research Lab
            (6, 350, 400, 'canyon', 0.30, 0.50, False),   # Canyon Entry
            (15, 350, 200, 'plateau', 0.25, 0.90, True),  # Central Hub
            (19, 250, 200, 'plain', 0.30, 0.80, True),    # Crossroads
            
            # North Area - High Altitude
            (3, 200, 500, 'crater', 0.90, 0.20, False),   # Deep Crater
            (4, 250, 450, 'plateau', 0.50, 0.95, True),   # North Peak
            (5, 350, 500, 'plateau', 0.40, 1.00, True),   # Observatory
            (7, 450, 450, 'plateau', 0.60, 0.80, True),   # East Ridge
            
            # South Area - Scientific Sites
            (11, 400, 100, 'plateau', 0.20, 0.85, True),  # South Base
            (12, 300, 50, 'canyon', 0.40, 0.45, False),   # Ancient River
            (13, 350, 100, 'canyon', 0.50, 0.50, False),  # Mineral Site
            (14, 200, 50, 'crater', 0.80, 0.30, False),   # Volcanic Vent
            
            # East Area - Plains
            (8, 500, 350, 'plain', 0.40, 0.70, True),     # Dust Plains
            (9, 500, 250, 'plain', 0.50, 0.30, False),    # Storm Zone
            (10, 450, 150, 'plain', 0.30, 0.60, True),    # Ice Deposit
            
            # West Area - Mixed Terrain
            (2, 150, 350, 'crater', 0.70, 0.40, False),   # Crater Edge
            (16, 150, 200, 'canyon', 0.35, 0.10, False),  # Water Cave
            (17, 100, 300, 'plain', 0.45, 0.75, True),    # West Outpost
            (18, 200, 400, 'canyon', 0.20, 0.40, False),  # Shelter Bay
        ]
        
        # Create and add stations
        for data in stations_data:
            station = MarsStation(
                id=data[0],
                position=(data[1], data[2]),
                terrain_type=data[3],
                radiation_level=data[4],
                solar_coverage=data[5],
                has_orbiter_visibility=data[6]
            )
            network.add_node(station)
        
        # Define hexagonal connections
        connections = [
            # Central hub connections
            (0, 1), (0, 6), (0, 15), (0, 19),
            (1, 6), (1, 7), (1, 15),
            (15, 19), (15, 13), (15, 10),
            
            # North area connections
            (3, 4), (4, 5), (4, 6), (4, 18),
            (5, 6), (5, 7),
            (6, 7), (6, 18),
            (7, 8),
            
            # South area connections
            (11, 12), (11, 13), (11, 10),
            (12, 13), (12, 14),
            (13, 14),
            
            # East area connections
            (8, 9), (9, 10),
            
            # West area connections
            (2, 3), (2, 17), (2, 18), (2, 19),
            (16, 17), (16, 14), (16, 19),
            (17, 18), (17, 19),
            (18, 3),
        ]
        
        for node1_id, node2_id in connections:
            network.add_link(node1_id, node2_id)
            
        return network
    
    def get_node_positions(self) -> Dict[int, Tuple[float, float]]:
        """Return positions of all stations for visualization."""
        return {node_id: station.position for node_id, station in self.nodes.items()}
    
    def get_node_colors(self, scenario: str = 'scientific') -> List[str]:
        """
        Return colors for nodes based on their properties and scenario.
        
        Args:
            scenario: Type of scenario ('scientific', 'solar', or 'communication')
        """
        colors = []
        for node_id in sorted(self.nodes.keys()):
            station = self.nodes[node_id]
            
            if scenario == 'scientific':
                # Color based on radiation level
                if station.radiation_level > 0.6:
                    colors.append('red')       # High radiation
                elif station.radiation_level > 0.3:
                    colors.append('orange')    # Medium radiation
                else:
                    colors.append('lightblue') # Low radiation
            
            elif scenario == 'solar':
                # Color based on solar coverage
                if station.solar_coverage < 0.4:
                    colors.append('red')       # Low solar
                elif station.solar_coverage < 0.7:
                    colors.append('orange')    # Medium solar
                else:
                    colors.append('lightblue') # High solar
            
            elif scenario == 'communication':
                # Color based on orbiter visibility
                if station.has_orbiter_visibility:
                    colors.append('lightblue')  # Has visibility
                else:
                    colors.append('red')        # No visibility
                    
        return colors
    
    def calculate_path_cost(self, path: List[int]) -> float:
        """
        Calculate the total cost of a path.
        
        Args:
            path: List of node IDs representing the path
            
        Returns:
            float: Total cost of the path
        """
        if not path or len(path) < 2:
            return 0.0
            
        total_cost = 0.0
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i+1])
            if edge_data and 'weight' in edge_data:
                total_cost += edge_data['weight']
                
        return total_cost
    
    def calculate_path_radiation(self, path: List[int]) -> float:
        """Calculate maximum radiation exposure along a path."""
        if not path:
            return 0.0
        return max(self.nodes[node_id].radiation_level for node_id in path)
    
    def calculate_path_solar(self, path: List[int]) -> float:
        """Calculate minimum solar coverage along a path."""
        if not path:
            return 0.0
        return min(self.nodes[node_id].solar_coverage for node_id in path)
    
    def count_visibility_nodes(self, path: List[int]) -> int:
        """Count nodes with orbiter visibility in a path."""
        if not path:
            return 0
        return sum(1 for node_id in path if self.nodes[node_id].has_orbiter_visibility)
    
    def validate_path(self, path: List[int]) -> bool:
        """
        Validate that a path is valid (consecutive nodes are connected).
        
        Args:
            path: List of node IDs
            
        Returns:
            bool: True if path is valid
        """
        if not path:
            return False
            
        for i in range(len(path) - 1):
            if not self.graph.has_edge(path[i], path[i+1]):
                return False
        return True
    
    def validate_path_constraints(self, 
                                  path: List[int],
                                  scenario: str,
                                  constraints: Dict) -> bool:
        """
        Validate path against scenario-specific constraints.
        
        Args:
            path: List of node IDs
            scenario: Type of scenario
            constraints: Dictionary of constraint values
            
        Returns:
            bool: True if constraints are satisfied
        """
        if not self.validate_path(path):
            return False
            
        if scenario == 'scientific':
            # Check radiation constraints
            max_radiation = constraints.get('max_radiation', 0.6)
            for node_id in path:
                if self.nodes[node_id].radiation_level > max_radiation:
                    return False
                    
        elif scenario == 'solar':
            # Check solar coverage constraints
            # Start and goal are fixed points and may be exempt from this constraint.
            min_solar = constraints.get('min_solar', 0.5)
            intermediate_nodes = path[1:-1]
            for node_id in intermediate_nodes:
                if self.nodes[node_id].solar_coverage < min_solar:
                    return False
                    
        elif scenario == 'communication':
            # Check visibility constraints (at least 1 visible node every 2 hops)
            max_blind_hops = constraints.get('max_blind_hops', 2)
            blind_count = 0
            for node_id in path:
                if self.nodes[node_id].has_orbiter_visibility:
                    blind_count = 0
                else:
                    blind_count += 1
                    if blind_count > max_blind_hops:
                        return False
            
        return True
    
    def get_edge_weights(self) -> Dict[Tuple[int, int], float]:
        """Return all edge weights."""
        return nx.get_edge_attributes(self.graph, 'weight')