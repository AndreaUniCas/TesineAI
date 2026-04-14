"""
Mars Station node class for Mars Rover exploration network.
"""
from dataclasses import dataclass
from typing import Tuple, Dict, Optional


@dataclass
class MarsStation:
    """
    Represents a Mars exploration station/node.
    
    Attributes:
        id: Unique identifier for the station
        position: (x, y) coordinates on the Mars surface map
        terrain_type: Type of terrain ('crater', 'plain', 'canyon', 'plateau')
        radiation_level: Cosmic radiation level (0.0 - 1.0, higher = more dangerous)
        solar_coverage: Solar panel efficiency at this location (0.0 - 1.0)
        has_orbiter_visibility: Whether the station has line-of-sight to orbiter
    """
    id: int
    position: Tuple[float, float]
    terrain_type: str = 'plain'
    radiation_level: float = 0.3
    solar_coverage: float = 0.7
    has_orbiter_visibility: bool = True
    
    @property
    def x(self) -> float:
        """Get x coordinate."""
        return self.position[0]
    
    @property
    def y(self) -> float:
        """Get y coordinate."""
        return self.position[1]
    
    def distance_to(self, other: 'MarsStation') -> float:
        """Calculate Euclidean distance to another station."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
    
    def get_terrain_difficulty(self) -> float:
        """
        Get terrain difficulty multiplier based on terrain type.
        
        Returns:
            float: Difficulty multiplier (1.0 = normal, higher = harder)
        """
        terrain_factors = {
            'crater': 2.0,      # Very difficult terrain
            'canyon': 1.5,      # Moderately difficult
            'plain': 1.0,       # Easy terrain
            'plateau': 1.2      # Slightly elevated, moderate
        }
        return terrain_factors.get(self.terrain_type, 1.0)
    
    def get_traversal_cost(self, other: 'MarsStation') -> float:
        """
        Calculate the cost of traversing from this station to another.
        
        The cost considers:
        - Base distance (Euclidean)
        - Terrain difficulty of both stations
        
        Args:
            other: Destination station
            
        Returns:
            float: Traversal cost
        """
        base_distance = self.distance_to(other)
        
        # Average terrain difficulty
        avg_terrain = (self.get_terrain_difficulty() + other.get_terrain_difficulty()) / 2
        
        return base_distance * avg_terrain
    
    def get_radiation_risk(self, other: 'MarsStation') -> float:
        """
        Calculate radiation risk for a connection with another station.
        
        Returns the maximum radiation level between the two stations,
        as the rover will be exposed to the highest radiation on the path.
        
        Args:
            other: Other station
            
        Returns:
            float: Radiation risk (0.0 - 1.0)
        """
        return max(self.radiation_level, other.radiation_level)
    
    def get_solar_efficiency(self, other: 'MarsStation') -> float:
        """
        Calculate average solar efficiency for a connection.
        
        Args:
            other: Other station
            
        Returns:
            float: Average solar coverage (0.0 - 1.0)
        """
        return (self.solar_coverage + other.solar_coverage) / 2
    
    def to_dict(self) -> Dict:
        """Convert station attributes to dictionary."""
        return {
            'id': self.id,
            'position': self.position,
            'terrain_type': self.terrain_type,
            'radiation_level': self.radiation_level,
            'solar_coverage': self.solar_coverage,
            'has_orbiter_visibility': self.has_orbiter_visibility
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MarsStation':
        """Create station from dictionary data."""
        return cls(
            id=data['id'],
            position=data['position'],
            terrain_type=data.get('terrain_type', 'plain'),
            radiation_level=data.get('radiation_level', 0.3),
            solar_coverage=data.get('solar_coverage', 0.7),
            has_orbiter_visibility=data.get('has_orbiter_visibility', True)
        )
    
    def __repr__(self) -> str:
        return (f"MarsStation(id={self.id}, terrain='{self.terrain_type}', "
                f"radiation={self.radiation_level:.2f}, solar={self.solar_coverage:.2f}, "
                f"visibility={self.has_orbiter_visibility})")