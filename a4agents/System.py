import ray
from ray.dag import InputNode
from typing import Callable, Dict, Any, Union, Optional
from a4agents.SystemNode import SystemNode
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Ray
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

class System:
    """A robust AI agent system using Ray DAGs."""
    
    def __init__(self, name: str):
        self.name = name
        self.nodes = {}  # Stores agent nodes
        self.dag = None  # Stores DAG workflow
    
    def node(self, func: Optional[Callable] = None, *, name: Optional[str] = None, resources: Optional[Dict[str, Any]] = None):
        """
        Registers a function as a SystemNode in the Ray DAG, supporting both decorator and direct function call.

        Usage:
        - As a decorator: `@system.node(name="my_node")`
        - As a function: `system.node(my_function, name="my_node")`

        :param func: Function to register (optional for decorator usage).
        :param name: Unique name for the node.
        :param resources: Resource allocation for Ray DAG (e.g., {"num_cpus": 2, "num_gpus": 1}).
        :return: SystemNode instance (when used directly) or decorator (when used as `@system.node`).
        """
        
        def register_function(func: Callable):
            """Inner function to handle function registration logic."""
            if not callable(func):
                raise TypeError(f"Expected a callable function, got {type(func).__name__}")

            if not isinstance(name, str) or not name.strip():
                raise ValueError("Node name must be a non-empty string.")

            if name in self.nodes:
                raise ValueError(f"Node name '{name}' already exists in system '{self.name}'.")

            if any(node.func == func for node in self.nodes.values()):
                raise ValueError(f"Function '{func.__name__}' is already registered under a different name.")

            try:
                node = SystemNode(func, name, resources)
                self.nodes[name] = node
                logger.info(f"Node '{name}' added to System '{self.name}'.")
                return node
            except Exception as e:
                logger.error(f"Error creating node '{name}': {str(e)}")
                raise

        # If `func` is provided, it's being used directly: `system.node(my_function, name="my_node")`
        if func is not None:
            return register_function(func)

        # If `func` is None, return a decorator: `@system.node(name="my_node")`
        def decorator(func: Callable):
            return register_function(func)

        return decorator

    def add_nodes_from_dict(self, func_dict: Dict[str, Callable], resources: Optional[Dict[str, Any]] = None):
        """
        Adds multiple functions as SystemNodes from a dictionary.
        
        :param func_dict: Dictionary where keys are node names and values are functions.
        :param resources: Optional dictionary specifying resource allocation.
        :raises TypeError: If func_dict is not a dictionary or contains invalid keys/values.
        :raises ValueError: If any node name is already registered.
        """
        if not isinstance(func_dict, dict):
            raise TypeError("Expected a dictionary with {name: function} mapping, but got {type(func_dict).__name__}")

        for name, func in func_dict.items():
            if not isinstance(name, str) or not name.strip():
                logger.error(f"Invalid node name '{name}'. It must be a non-empty string.")
                raise TypeError(f"Node name '{name}' must be a non-empty string.")

            if not callable(func):
                logger.error(f"Invalid function '{name}': Expected a callable, but got {type(func).__name__}.")
                raise TypeError(f"Function '{name}' must be a callable function.")

            if name in self.nodes:
                logger.error(f"Duplicate node name '{name}'. Node names must be unique in system '{self.name}'.")
                raise ValueError(f"Node name '{name}' already exists in system '{self.name}'.")

            try:
                # Create and register the node
                node = SystemNode(func, name, resources)
                self.nodes[name] = node
                logger.info(f"Node '{name}' successfully added to System '{self.name}'.")
            except Exception as e:
                logger.error(f"Error adding node '{name}': {str(e)}")
                raise


    def workflow(self, workflow: Dict[str, list]):
        """
        Defines execution order in the system's DAG.

        Parameters:
        - workflow (dict): { "parent_node": ["child_node1", "child_node2", ...] }
        """
        try:
            dag = {}
            input_node = InputNode()

            for parent, children in workflow.items():
                if parent not in self.nodes:
                    raise ValueError(f"Node '{parent}' not registered in the system.")

                parent_node = self.nodes[parent]

                for child in children:
                    if child not in self.nodes:
                        raise ValueError(f"Node '{child}' not registered in the system.")
                    
                    child_node = self.nodes[child]
                    dag[child] = child_node.bind(parent_node.bind(input_node))

            self.dag = dag
        except Exception as e:
            raise RuntimeError(f"Error in workflow setup: {e}")

    def execute(self, start_node: str, data: Any):
        """
        Executes the workflow DAG starting from a specified node.

        Parameters:
        - start_node (str): Name of the starting node.
        - data (Any): Input data for execution.

        Returns:
        - The output of the final node execution.
        """
        if not self.dag:
            raise ValueError("No DAG defined. Call `workflow()` first.")

        if start_node not in self.dag:
            raise ValueError(f"Start node '{start_node}' not found in DAG.")

        try:
            return ray.get(self.dag[start_node].execute(data))
        except Exception as e:
            raise RuntimeError(f"Error executing DAG from '{start_node}': {e}")
