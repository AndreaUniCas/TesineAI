"""
BFS algorithm implementation for Mars Rover pathfinding.
"""

from typing import List, Dict, Set
from collections import deque
from src.models.network import MarsNetwork


class BFSPathfinder:
    """
    BFS algorithm for shortest path (by number of hops).
    
    This class implements Breadth-First Search for finding the path
    with the minimum number of edges (hops) between two nodes.
    
    Algorithm Overview:
    1. Initialize queue with start node
    2. Mark start as visited
    3. While queue is not empty:
       - Dequeue current node
       - If current is goal, reconstruct and return path
       - For each unvisited neighbor:
         * Mark as visited
         * Record came_from
         * Enqueue neighbor
    4. Return empty list if no path found
    
    Time Complexity: O(V + E)
    Space Complexity: O(V) for visited set and queue
    
    Note: BFS finds the shortest path by hop count, NOT by edge weight.
    For weighted shortest paths, use A* or Dijkstra's algorithm.
    
    Example Usage:
        network = MarsNetwork.create_mars_network()
        bfs = BFSPathfinder(network)
        path = bfs.find_path(start=0, goal=13)
    """

    def __init__(self, network: MarsNetwork):
        """
        Initialize BFS pathfinder with network.

        Args:
            network: The Mars exploration network to search
        """
        self.network = network
        self._nodes_expanded: int = 0

    def find_path(self, start: int, goal: int) -> List[int]:
        """
        Find shortest path (by hops) from start to goal using BFS.

        Args:
            start: Starting node ID
            goal: Goal node ID

        Returns:
            List[int]: List of node IDs representing the path from start to goal.
                      Empty list if no path exists.
        """
        self._nodes_expanded = 0

        # 1. Edge case: start is already the goal
        if start == goal:
            return [start]

        # 2. Initialize data structures
        queue: deque = deque([start])
        visited: Set[int] = {start}
        came_from: Dict[int, int] = {}

        # 3. Main BFS loop
        while queue:
            current = queue.popleft()
            self._nodes_expanded += 1

            for neighbor in self.network.graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current

                    if neighbor == goal:
                        return self._reconstruct_path(came_from, goal)

                    queue.append(neighbor)

        # 4. No path found
        return []

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
