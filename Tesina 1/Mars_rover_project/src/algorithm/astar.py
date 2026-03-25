"""
A* algorithm implementation for Mars Rover pathfinding.
"""

from typing import List, Dict, Set, Optional
from queue import PriorityQueue
from src.models.network import MarsNetwork
from src.algorithm.cost_functions import BaseHeuristic, EuclideanHeuristic


class AStarPathfinder:
    """
    A* algorithm for optimal pathfinding on Mars.
    
    This class implements the A* search algorithm using a priority queue
    for efficient node selection based on f(n) = g(n) + h(n).
    
    Algorithm Overview:
    1. Initialize open_set with start node
    2. g_score[start] = 0, f_score[start] = heuristic(start, goal)
    3. While open_set is not empty:
       - Extract node with lowest f_score
       - If node is goal, reconstruct and return path
       - For each neighbor:
         * Calculate tentative g_score
         * If better than current, update scores and add to open_set
    4. Return empty list if no path found
    
    Time Complexity: O(E) best case, O(V²) worst case
    Space Complexity: O(V) for storing scores and came_from
    
    Example Usage:
        network = MarsNetwork.create_mars_network()
        astar = AStarPathfinder(network)
        path = astar.find_path(start=0, goal=13)
    """

    def __init__(self, network: MarsNetwork, heuristic: Optional[BaseHeuristic] = None):
        """
        Initialize A* pathfinder with network.

        Args:
            network: The Mars exploration network to search
            heuristic: Optional custom heuristic (defaults to EuclideanHeuristic)
        """
        self.network = network
        self.heuristic = heuristic or EuclideanHeuristic(network)
        self._nodes_expanded: int = 0

    def find_path(self, start: int, goal: int) -> List[int]:
        """
        Find optimal path from start to goal using A* algorithm.

        Args:
            start: Starting node ID
            goal: Goal node ID

        Returns:
            List[int]: List of node IDs representing the path from start to goal.
                      Empty list if no path exists.
        """
        # Reset expansion counter for this search
        self._nodes_expanded = 0

        # 1. Initialize data structures
        open_set = PriorityQueue()
        open_set_hash: Set[int] = set()
        came_from: Dict[int, int] = {}
        g_score: Dict[int, float] = {}
        f_score: Dict[int, float] = {}

        # 2. Initialize start and goal nodes
        g_score[start] = 0
        f_score[start] = self._heuristic(start, goal)
        g_score[goal] = float('inf')
        f_score[goal] = float('inf')

        # Add start node to open set
        open_set.put((f_score[start], start))
        open_set_hash.add(start)

        # 3. Main loop: process nodes until the open set is exhausted
        while not open_set.empty():
            # a. Get node with lowest f_score
            _, current = open_set.get()

            # b. Remove from open_set_hash (mark as closed)
            open_set_hash.discard(current)

            # Count this node as expanded
            self._nodes_expanded += 1

            # c. Goal check: reconstruct and return the path
            if current == goal:
                return self._reconstruct_path(came_from, current)

            # d. Explore each neighbour of the current node
            for neighbor in self.network.graph.neighbors(current):
                # Use scenario-specific edge cost if available
                edge_weight = self.heuristic.get_edge_cost(current, neighbor)
                tentative_g = g_score[current] + edge_weight

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)

                    if neighbor not in open_set_hash:
                        open_set.put((f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

        # 4. Open set exhausted — no path exists
        return []

    def _heuristic(self, node: int, goal: int) -> float:
        """
        Delegate heuristic calculation to the injected heuristic object.

        Args:
            node: Current node ID
            goal: Goal node ID

        Returns:
            float: Estimated cost from node to goal
        """
        return self.heuristic.calculate(node, goal)

    def _reconstruct_path(self, came_from: Dict[int, int], current: int) -> List[int]:
        """
        Reconstruct the path from start to current node.

        Args:
            came_from: Dictionary mapping each node to its predecessor
            current: The goal node (end of path)

        Returns:
            List[int]: Complete path from start to goal
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def get_nodes_expanded(self) -> int:
        """
        Return the number of nodes expanded during the last search.

        Returns:
            int: Number of nodes expanded
        """
        return self._nodes_expanded
