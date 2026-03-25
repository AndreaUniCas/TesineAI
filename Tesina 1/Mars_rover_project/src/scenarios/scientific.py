"""
Scientific Sample Collection scenario implementation.
Goal: Find optimal path to collect samples while minimizing radiation exposure.
"""

from typing import List, Optional, Dict
from src.models.network import MarsNetwork
from src.algorithm.astar import AStarPathfinder
from src.algorithm.bfs import BFSPathfinder
from src.algorithm.cost_functions import RadiationHeuristic


def solve_scientific_scenario(network: MarsNetwork,
                               algorithm: str = 'astar',
                               start: int = 0,
                               goal: int = 13,
                               constraints: Dict = None) -> Optional[List[int]]:
    """
    Find optimal path for scientific sample collection.
    
    The rover must travel from Base Camp to the Mineral Site while
    minimizing exposure to cosmic radiation.
    
    Args:
        network: The Mars network
        algorithm: Pathfinding algorithm to use ('astar' or 'bfs')
        start: Starting node ID (default: 0 - Base Camp)
        goal: Goal node ID (default: 13 - Mineral Site)
        constraints: Dictionary of constraints including:
                    - max_radiation: Maximum allowed radiation per node (default: 0.6)
                    
    Returns:
        Optional[List[int]]: Path as list of node IDs if found
    """
    constraints = constraints or {}

    # 1. Build the radiation-aware heuristic
    heuristic = RadiationHeuristic(network, constraints)

    # 2. Initialize the chosen pathfinding algorithm
    if algorithm == 'astar':
        pathfinder = AStarPathfinder(network, heuristic)
    else:
        # BFS ignores edge weights; heuristic has no effect, but we keep the
        # interface consistent
        pathfinder = BFSPathfinder(network)

    # 3. Find the path
    path = pathfinder.find_path(start, goal)

    # 4. Validate and return
    if validate_scientific_path(network, path, constraints):
        return path

    return None


def validate_scientific_path(network: MarsNetwork,
                              path: List[int],
                              constraints: Dict) -> bool:
    """
    Validate path for scientific sample collection scenario.

    Args:
        network: The Mars network
        path: Proposed path solution
        constraints: Scenario constraints

    Returns:
        bool: True if path is valid
    """
    if not path:
        return False

    max_radiation = constraints.get('max_radiation', 0.6)

    for node_id in path:
        station = network.nodes[node_id]
        if station.radiation_level > max_radiation:
            return False

    return network.validate_path(path)


def analyze_scientific_path(network: MarsNetwork, path: List[int]) -> Dict:
    """
    Analyze the scientific path metrics.

    Args:
        network: The Mars network
        path: The found path

    Returns:
        Dict: Analysis metrics
    """
    if not path:
        return {'error': 'No path provided'}

    return {
        'path_length': len(path),
        'total_cost': network.calculate_path_cost(path),
        'max_radiation': network.calculate_path_radiation(path),
        'nodes_visited': path,
        'terrain_types': [network.nodes[n].terrain_type for n in path]
    }
