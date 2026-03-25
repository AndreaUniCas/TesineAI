"""
Solar Power Route scenario implementation.
Goal: Find path that maximizes solar exposure for battery charging.
"""

from typing import List, Optional, Dict
from src.models.network import MarsNetwork
from src.algorithm.astar import AStarPathfinder
from src.algorithm.bfs import BFSPathfinder
from src.algorithm.cost_functions import SolarHeuristic


def solve_solar_scenario(network: MarsNetwork,
                         algorithm: str = 'astar',
                         start: int = 16,
                         goal: int = 5,
                         constraints: Dict = None) -> Optional[List[int]]:
    """
    Find optimal path for solar-powered travel.
    
    The rover has low battery and must travel from Water Cave to the
    Observatory charging station, preferring paths with high solar exposure.
    
    Args:
        network: The Mars network
        algorithm: Pathfinding algorithm to use ('astar' or 'bfs')
        start: Starting node ID (default: 16 - Water Cave, low battery)
        goal: Goal node ID (default: 5 - Observatory, charging station)
        constraints: Dictionary of constraints including:
                    - min_solar: Minimum required solar coverage per node (default: 0.5)
                    
    Returns:
        Optional[List[int]]: Path as list of node IDs if found
    """
    constraints = constraints or {}

    # 1. Build the solar-aware heuristic
    heuristic = SolarHeuristic(network, constraints)

    # 2. Initialize the chosen pathfinding algorithm
    if algorithm == 'astar':
        pathfinder = AStarPathfinder(network, heuristic)
    else:
        pathfinder = BFSPathfinder(network)

    # 3. Find the path
    path = pathfinder.find_path(start, goal)

    # 4. Validate and return
    if validate_solar_path(network, path, constraints):
        return path

    return None


def validate_solar_path(network: MarsNetwork,
                        path: List[int],
                        constraints: Dict) -> bool:
    """
    Validate path for solar power route scenario.

    Args:
        network: The Mars network
        path: Proposed path solution
        constraints: Scenario constraints

    Returns:
        bool: True if path is valid
    """
    if not path:
        return False

    min_solar = constraints.get('min_solar', 0.5)
    
    # Skip start and goal: they are fixed and may not meet the solar constraint
    intermediate_nodes = path[1:-1]
    for node_id in intermediate_nodes:
        station = network.nodes[node_id]
        if station.solar_coverage < min_solar:
            return False

    return network.validate_path(path)


def analyze_solar_path(network: MarsNetwork, path: List[int]) -> Dict:
    """
    Analyze the solar path metrics.

    Args:
        network: The Mars network
        path: The found path

    Returns:
        Dict: Analysis metrics
    """
    if not path:
        return {'error': 'No path provided'}

    solar_values = [network.nodes[n].solar_coverage for n in path]

    return {
        'path_length': len(path),
        'total_cost': network.calculate_path_cost(path),
        'min_solar': min(solar_values),
        'avg_solar': sum(solar_values) / len(solar_values),
        'total_solar_exposure': sum(solar_values),
        'nodes_visited': path
    }
