import ray
import logging
from typing import Callable, Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

class SystemNode:
    """Encapsulates a function as a Ray Task for optimized execution."""

    def __init__(self, func: Callable, name: str, resources: Optional[Dict[str, Any]] = None):
        """
        Initializes the SystemNode as a Ray Task.

        :param func: Function to be wrapped as a Ray task.
        :param name: Unique name for the node.
        :param resources: Resource allocation for Ray (e.g., {"num_cpus": 2, "num_gpus": 1}).
        """
        self.name = name
        self.resources = resources or {}
        self.func = func

        try:
            # Create a Ray task with resource constraints if provided
            self.task = ray.remote(**self.resources)(func)
            logger.info(f"SystemNode '{self.name}' initialized successfully as a Ray Task.")
        except Exception as e:
            logger.error(f"Error initializing SystemNode '{self.name}': {str(e)}")
            raise

    def execute(self, *args, **kwargs) -> ray.ObjectRef:
        """
        Executes the function within Ray.

        :param args: Positional arguments for the function.
        :param kwargs: Keyword arguments for the function.
        :return: Ray ObjectRef (use ray.get() to retrieve the actual result).
        """
        try:
            result = self.task.remote(*args, **kwargs)  # Corrected execution
            logger.info(f"Execution started for SystemNode '{self.name}'.")
            return result  # Returns Ray ObjectRef
        except Exception as e:
            logger.error(f"Error executing SystemNode '{self.name}': {str(e)}")
            raise




class Router:
    def __init__(self, router_id: Optional[str] = None):
        """
        Initializes a Router instance to manage dynamic workflow routing.

        Args:
            router_id (Optional[str]): An optional unique identifier for the router.
        """
        self.router_id = router_id
        self.mapping_dict: Dict[str, Dict[str, str]] = {}  # {from_node: {output: to_node}}

    def add_routes(self, from_node: str, to_node: Optional[str] = None, mapping: Optional[Dict[str, str]] = None):
        """
        Adds a routing rule. Either `to_node` (direct edge) or `mapping` (output-based mapping) must be provided.

        Args:
            from_node (str): The node where routing starts.
            to_node (Optional[str]): The direct node to transition to.
            mapping (Optional[Dict[str, str]]): A dictionary mapping outputs to next nodes.

        Raises:
            ValueError: If `from_node` already exists in mappings.
            ValueError: If neither `to_node` nor `mapping` is provided.
        """
        if from_node in self.mapping_dict:
            raise ValueError(f"Cannot add a route for '{from_node}', it already maps to {self.mapping_dict[from_node]}.")

        if to_node:
            self.mapping_dict[from_node] = {"default": to_node}  # Treat single transitions as a default case.
        elif mapping:
            self.mapping_dict[from_node] = mapping
        else:
            raise ValueError(f"Must provide either 'to_node' or 'mapping' for route '{from_node}'.")

    def get_next_node(self, from_node: str, output: Optional[Union[str, List[str]]] = None) -> List[str]:
        """
        Determines the next node(s) based on direct edges or output mapping.

        Args:
            from_node (str): The current node.
            output (Optional[Union[str, List[str]]]): The output(s) for mapped routing.

        Returns:
            List[str]: A list of next node(s) based on the mapping.

        Raises:
            ValueError: If `from_node` is unknown or if required output is missing.
        """
        if from_node not in self.mapping_dict:
            raise ValueError(f"Unknown node '{from_node}'. It must be added using 'add_route()'.")

        next_mapping = self.mapping_dict[from_node]

        # Handle direct transition (if default key exists)
        if "default" in next_mapping:
            return [next_mapping["default"]]

        # Ensure output is valid for mapped transitions
        if output is None:
            raise ValueError(f"Output required for mapped route '{from_node}', but none was provided.")

        # Convert single string output to a list for consistent handling
        if isinstance(output, str):
            output = [output]

        if not isinstance(output, list) or not all(isinstance(o, str) for o in output):
            raise TypeError(f"Invalid output format for node '{from_node}'. Expected str or List[str].")

        next_nodes = []
        for val in output:
            if val not in next_mapping:
                raise ValueError(f"Invalid output '{val}' for node '{from_node}'. Expected one of {list(next_mapping.keys())}.")
            next_nodes.append(next_mapping[val])

        return next_nodes

    def get_metadata(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Returns metadata about all defined routes and mappings."""
        return {
            "router_id": self.router_id,
            "mappings": self.mapping_dict
        }

