"""
Cost and heuristic functions for Mars Rover pathfinding.

This module provides different heuristic calculation strategies that can be
used to customize pathfinding for different scenarios (Scientific, Solar, Communication).
"""

from abc import ABC, abstractmethod
from typing import Dict
from src.models.network import MarsNetwork
from src.models.node import MarsStation


class BaseHeuristic(ABC):
    """
    Abstract base class for heuristic function implementations.
    
    Heuristics determine how the estimated cost to the goal is calculated
    for different optimization scenarios. Each scenario may prioritize
    different factors (radiation avoidance, solar exposure, communication).
    
    Students can extend this class to create scenario-specific heuristics.
    """
    
    def __init__(self, network: MarsNetwork):
        """
        Initialize heuristic with network.
        
        Args:
            network: The Mars network
        """
        self.network = network
    
    @abstractmethod
    def calculate(self, node_id: int, goal_id: int) -> float:
        """
        Calculate heuristic estimate from node to goal.
        
        Args:
            node_id: Current node ID
            goal_id: Goal node ID
            
        Returns:
            float: Estimated cost from node to goal
        """
        pass
    
    def get_edge_cost(self, node1_id: int, node2_id: int) -> float:
        """
        Get the modified edge cost for this heuristic.
        
        Can be overridden to provide scenario-specific edge costs.
        
        Args:
            node1_id: First node ID
            node2_id: Second node ID
            
        Returns:
            float: Edge cost
        """
        edge_data = self.network.graph.get_edge_data(node1_id, node2_id)
        if edge_data:
            return edge_data.get('weight', float('inf'))
        return float('inf')


class EuclideanHeuristic(BaseHeuristic):
    """
    Basic Euclidean distance heuristic.
    
    This is the simplest admissible heuristic for graphs embedded in
    2D space. It never overestimates the actual cost if edge weights
    are based on Euclidean distance.
    """
    
    def calculate(self, node_id: int, goal_id: int) -> float:
        """Calculate Euclidean distance to goal."""
        node = self.network.nodes[node_id]
        goal = self.network.nodes[goal_id]
        return node.distance_to(goal)


class RadiationHeuristic(BaseHeuristic):
    """
    Heuristic for Scientific Sample Collection scenario.
    
    Prioritizes paths with low radiation exposure.
    Penalizes nodes with high radiation levels.
    """
    
    def __init__(self, network: MarsNetwork, constraints: Dict = None):
        """
        Initialize Radiation heuristic.
        
        Args:
            network: The Mars network
            constraints: Optional constraints with max_radiation
        """
        super().__init__(network)
        self.constraints = constraints or {}
        self.max_radiation = self.constraints.get('max_radiation', 0.6)
    
    def calculate(self, node_id: int, goal_id: int) -> float:
        """
        Calculate heuristic considering radiation levels.
        
        TODO: Student Implementation
        
        Consider:
        - Base Euclidean distance
        - Radiation level at current node
        - Higher radiation = higher cost multiplier
        
        Formula suggestion:
        h(n) = distance(n, goal) * (1 + radiation_level(n))
        """
        node = self.network.nodes[node_id]
        goal = self.network.nodes[goal_id]
        
        # Base distance
        distance = node.distance_to(goal)
        
        # Radiation penalty factor
        radiation_factor = 1.0 + node.radiation_level
        
        return distance * radiation_factor
    
    def get_edge_cost(self, node1_id: int, node2_id: int) -> float:
        """
        Get modified edge cost considering radiation.
        
        TODO: Student can enhance this
        """
        base_cost = super().get_edge_cost(node1_id, node2_id)
        
        # Add radiation penalty
        node1 = self.network.nodes[node1_id]
        node2 = self.network.nodes[node2_id]
        radiation_risk = max(node1.radiation_level, node2.radiation_level)
        
        return base_cost * (1 + radiation_risk)


class SolarHeuristic(BaseHeuristic):
    """
    Heuristic for Solar Power Route scenario.
    
    Prioritizes paths with high solar coverage.
    Penalizes nodes with low solar exposure.
    """
    
    def __init__(self, network: MarsNetwork, constraints: Dict = None):
        """
        Initialize Solar heuristic.
        
        Args:
            network: The Mars network
            constraints: Optional constraints with min_solar
        """
        super().__init__(network)
        self.constraints = constraints or {}
        self.min_solar = self.constraints.get('min_solar', 0.5)
    
    def calculate(self, node_id: int, goal_id: int) -> float:
        """
        Calculate heuristic considering solar coverage.
        
        TODO: Student Implementation
        
        Consider:
        - Base Euclidean distance
        - Solar coverage at current node
        - Lower solar = higher cost (inverted)
        
        Formula suggestion:
        h(n) = distance(n, goal) * (2.0 - solar_coverage(n))
        """
        node = self.network.nodes[node_id]
        goal = self.network.nodes[goal_id]
        
        # Base distance
        distance = node.distance_to(goal)
        
        # Solar penalty (inverted: low solar = high cost)
        solar_factor = 2.0 - node.solar_coverage
        
        return distance * solar_factor
    
    def get_edge_cost(self, node1_id: int, node2_id: int) -> float:
        """
        Get modified edge cost considering solar coverage.
        
        TODO: Student can enhance this
        """
        base_cost = super().get_edge_cost(node1_id, node2_id)
        
        # Add solar penalty (low solar = high cost)
        node1 = self.network.nodes[node1_id]
        node2 = self.network.nodes[node2_id]
        avg_solar = (node1.solar_coverage + node2.solar_coverage) / 2
        
        return base_cost * (2.0 - avg_solar)


class CommunicationHeuristic(BaseHeuristic):
    """
    Heuristic for Communication Relay scenario.
    
    Prioritizes paths that maintain orbiter visibility.
    Penalizes nodes without line-of-sight to orbiter.
    """
    
    def __init__(self, network: MarsNetwork, constraints: Dict = None):
        """
        Initialize Communication heuristic.
        
        Args:
            network: The Mars network
            constraints: Optional constraints with max_blind_hops
        """
        super().__init__(network)
        self.constraints = constraints or {}
        self.max_blind_hops = self.constraints.get('max_blind_hops', 2)
    
    def calculate(self, node_id: int, goal_id: int) -> float:
        """
        Calculate heuristic considering orbiter visibility.
        
        TODO: Student Implementation
        
        Consider:
        - Base Euclidean distance
        - Visibility at current node
        - No visibility = higher cost
        
        Formula suggestion:
        h(n) = distance(n, goal) * (1.5 if not visible else 0.8)
        """
        node = self.network.nodes[node_id]
        goal = self.network.nodes[goal_id]
        
        # Base distance
        distance = node.distance_to(goal)
        
        # Visibility factor
        if node.has_orbiter_visibility:
            visibility_factor = 0.8  # Prefer visible nodes
        else:
            visibility_factor = 1.5  # Penalize blind nodes
        
        return distance * visibility_factor
    
    def get_edge_cost(self, node1_id: int, node2_id: int) -> float:
        """
        Get modified edge cost considering visibility.
        
        TODO: Student can enhance this
        """
        base_cost = super().get_edge_cost(node1_id, node2_id)
        
        # Add visibility penalty
        node2 = self.network.nodes[node2_id]
        
        if not node2.has_orbiter_visibility:
            return base_cost * 1.5
        return base_cost * 0.8


def get_heuristic_for_scenario(network: MarsNetwork, 
                               scenario: str, 
                               constraints: Dict = None) -> BaseHeuristic:
    """
    Factory function to get the appropriate heuristic for a scenario.
    
    Args:
        network: The Mars network
        scenario: Scenario type ('scientific', 'solar', 'communication')
        constraints: Optional scenario constraints
        
    Returns:
        BaseHeuristic: Appropriate heuristic instance
    """
    heuristics = {
        'scientific': RadiationHeuristic,
        'solar': SolarHeuristic,
        'communication': CommunicationHeuristic
    }
    
    heuristic_class = heuristics.get(scenario, EuclideanHeuristic)
    return heuristic_class(network, constraints)