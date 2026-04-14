import argparse
import logging
from typing import List, Dict, Tuple, Optional
import sys

sys.dont_write_bytecode = True

from src.models.network import MarsNetwork
from src.utils.visualization import NetworkVisualizer
from src.scenarios.scientific import solve_scientific_scenario
from src.scenarios.solar import solve_solar_scenario
from src.scenarios.communication import solve_communication_scenario


def setup_logging() -> None:
    """Configure logging for the application."""
    class CustomFormatter(logging.Formatter):
        """Custom formatter with colors and symbols"""
        grey = "\x1b[38;20m"
        blue = "\x1b[34;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        green = "\x1b[32;20m"
        reset = "\x1b[0m"

        def __init__(self):
            super().__init__()
            self.FORMATS = {
                logging.DEBUG: self.grey + "🔍 DEBUG: %(message)s" + self.reset,
                logging.INFO: self.blue + "ℹ️  %(message)s" + self.reset,
                logging.WARNING: self.yellow + "⚠️  WARNING: %(message)s" + self.reset,
                logging.ERROR: self.red + "❌ ERROR: %(message)s" + self.reset,
                logging.CRITICAL: self.bold_red + "🚨 CRITICAL: %(message)s" + self.reset
            }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.handlers = []
    logger.addHandler(ch)


def get_scenario_data(scenario_type: str) -> dict:
    """Get predefined data for each scenario."""
    scenarios = {
        'scientific': {
            'description': 'Scientific Sample Collection - Minimize radiation exposure',
            'default_start': 0,
            'default_goal': 13,
            'constraints': {
                'max_radiation': 0.6
            }
        },
        'solar': {
            'description': 'Solar Power Route - Maximize solar exposure for battery charging',
            'default_start': 16,
            'default_goal': 5,
            'constraints': {
                'min_solar': 0.5
            }
        },
        'communication': {
            'description': 'Communication Relay - Maintain orbiter visibility',
            'default_start': 3,
            'default_goal': 11,
            'constraints': {
                'max_blind_hops': 2
            }
        }
    }
    return scenarios.get(scenario_type, None)


def validate_path(network: MarsNetwork, path: List[int], 
                 scenario: str, constraints: Dict) -> bool:
    """Validate if path meets scenario constraints."""
    if not path:
        return False
    return network.validate_path_constraints(path, scenario, constraints)


def main():
    parser = argparse.ArgumentParser(
        description='Mars Rover Exploration Network - A* and BFS Pathfinding',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--scenario', 
        type=str, 
        choices=['scientific', 'solar', 'communication'],
        required=True,
        help='''Select scenario to solve:
scientific    - Find path minimizing radiation exposure
solar         - Find path maximizing solar coverage
communication - Find path maintaining orbiter visibility'''
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['astar', 'bfs'],
        default='astar',
        help='Select pathfinding algorithm to use (default: astar)'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=None,
        help='Start node ID (uses scenario default if not specified)'
    )
    parser.add_argument(
        '--goal',
        type=int,
        default=None,
        help='Goal node ID (uses scenario default if not specified)'
    )
    
    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Create and display initial network
        network = MarsNetwork.create_mars_network()
        logger.info("🚀 Initializing Mars Rover Exploration Network...")
        visualizer = NetworkVisualizer()
        visualizer.display_initial_network(network)
        
        # Get scenario data
        scenario_data = get_scenario_data(args.scenario)
        if not scenario_data:
            raise ValueError(f"Invalid scenario selection: {args.scenario}")
        
        # Determine start and goal nodes
        start = args.start if args.start is not None else scenario_data['default_start']
        goal = args.goal if args.goal is not None else scenario_data['default_goal']
        
        # Validate node IDs
        if start not in network.nodes:
            raise ValueError(f"Start node {start} does not exist in network")
        if goal not in network.nodes:
            raise ValueError(f"Goal node {goal} does not exist in network")
        
        logger.info(f"🎯 Running {args.scenario} scenario: {scenario_data['description']}")
        logger.info(f"📍 Start: Station {start} ({network.nodes[start].terrain_type})")
        logger.info(f"🏁 Goal: Station {goal} ({network.nodes[goal].terrain_type})")
        logger.info(f"🔧 Algorithm: {args.algorithm.upper()}")
        
        # Solve the selected scenario
        path = None
        if args.scenario == 'scientific':
            path = solve_scientific_scenario(
                network,
                args.algorithm,
                start,
                goal,
                scenario_data['constraints']
            )
        elif args.scenario == 'solar':
            path = solve_solar_scenario(
                network,
                args.algorithm,
                start,
                goal,
                scenario_data['constraints']
            )
        elif args.scenario == 'communication':
            path = solve_communication_scenario(
                network,
                args.algorithm,
                start,
                goal,
                scenario_data['constraints']
            )
        
        if path:
            # Validate the solution
            if validate_path(network, path, args.scenario, scenario_data['constraints']):
                # Calculate and log metrics
                total_cost = network.calculate_path_cost(path)
                path_str = ' → '.join([f'S{n}' for n in path])
                
                logger.info(f"✅ Path found! {path_str}")
                logger.info(f"📊 Total cost: {total_cost:.2f}")
                logger.info(f"📏 Path length: {len(path)} nodes")
                
                # Visualize the solution
                visualizer.plot_scenario(
                    network,
                    scenario_type=args.scenario,
                    path=path,
                    start=start,
                    goal=goal
                )
                visualizer.show_plot()
            else:
                logger.error("Path found but doesn't meet required constraints!")
        else:
            logger.error("No valid path found - Check your implementation!")
        
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())