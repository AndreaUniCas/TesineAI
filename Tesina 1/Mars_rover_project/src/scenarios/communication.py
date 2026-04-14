"""
Communication Relay scenario implementation.
Goal: Find path that maintains orbiter visibility for data transmission.
"""

from typing import List, Optional, Dict
from src.models.network import MarsNetwork
from src.algorithm.astar import AStarPathfinder
from src.algorithm.bfs import BFSPathfinder
from src.algorithm.cost_functions import CommunicationHeuristic


def solve_communication_scenario(network: MarsNetwork,
                                  algorithm: str = 'astar',
                                  start: int = 3,
                                  goal: int = 11,
                                  constraints: Dict = None) -> Optional[List[int]]:
    """
    Find optimal path maintaining communication with orbiter.
    
    The rover is stuck in Deep Crater without communication and must
    reach South Base while maintaining contact with the orbiter as much
    as possible.
    
    Args:
        network: The Mars network
        algorithm: Pathfinding algorithm to use ('astar' or 'bfs')
        start: Starting node ID (default: 3 - Deep Crater, no signal)
        goal: Goal node ID (default: 11 - South Base, main antenna)
        constraints: Dictionary of constraints including:
                    - max_blind_hops: Max consecutive nodes without visibility (default: 2)
                    
    Returns:
        Optional[List[int]]: Path as list of node IDs if found
    """
    constraints = constraints or {}

    # 1. Build the communication-aware heuristic
    heuristic = CommunicationHeuristic(network, constraints)

    # 2. Initialize the chosen pathfinding algorithm
    if algorithm == 'astar':
        pathfinder = AStarPathfinder(network, heuristic)
    else:
        pathfinder = BFSPathfinder(network)

    # 3. Find the path
    path = pathfinder.find_path(start, goal)

    # 4. Validate and return
    if validate_communication_path(network, path, constraints):
        return path

    return None


def validate_communication_path(network: MarsNetwork,
                                 path: List[int],
                                 constraints: Dict) -> bool:
    """
    Validate path for communication relay scenario.

    Args:
        network: The Mars network
        path: Proposed path solution
        constraints: Scenario constraints

    Returns:
        bool: True if path is valid
    """
    if not path:
        return False

    max_blind_hops = constraints.get('max_blind_hops', 2)

    # Count consecutive nodes without orbiter visibility
    blind_count = 0
    for node_id in path:
        station = network.nodes[node_id]
        if station.has_orbiter_visibility:
            blind_count = 0  # Reset when we have visibility
        else:
            blind_count += 1
            if blind_count > max_blind_hops:
                return False

    return network.validate_path(path)


def analyze_communication_path(network: MarsNetwork, path: List[int]) -> Dict:
    """
    Analyze the communication path metrics.

    Args:
        network: The Mars network
        path: The found path

    Returns:
        Dict: Analysis metrics
    """
    if not path:
        return {'error': 'No path provided'}

    visibility_count = network.count_visibility_nodes(path)
    visibility_ratio = visibility_count / len(path)

    # Calculate maximum consecutive blind hops
    max_blind = 0
    current_blind = 0
    for node_id in path:
        if network.nodes[node_id].has_orbiter_visibility:
            max_blind = max(max_blind, current_blind)
            current_blind = 0
        else:
            current_blind += 1
    max_blind = max(max_blind, current_blind)

    return {
        'path_length': len(path),
        'total_cost': network.calculate_path_cost(path),
        'nodes_with_visibility': visibility_count,
        'visibility_ratio': visibility_ratio,
        'max_consecutive_blind': max_blind,
        'nodes_visited': path
    }
