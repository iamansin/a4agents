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
        self.resources = resources
        self.func = func

        try:
            # Create a Ray task with resource constraints if provided
            self.task = ray.remote(**self.resources)(func) if resources else ray.remote(func)
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


class DAGExecuter:
    def __init__(self,system):
        """Initialize an empty DAG structure."""
        self._system = weakref.ref(system)
        self._graph = self._system()._dag

    @ray.remote
    def execute_node(self, name: str, *args, **kwargs) -> Any:
        """
        Execute a single node in the workflow.
        Uses `workflow.continuation` to dynamically decide the next node.
        :param name: Name of the node to execute.
        """
        if name not in self._system()._nodes:
            raise ValueError(f"Node '{name}' is not in the DAG.")
        
        try:
            result = self._system._graph.nodes[name]["func_ref"](*args, **kwargs)
            next_nodes = self._graph.successors(name)

            if not next_nodes:
                return result  # End of DAG, return the final result
            
            # Continue with the next node(s) dynamically
            if len(next_nodes) == 1:
                return workflow.continuation(self.execute_node.bind(next_nodes[0], result))
            else:
                return [workflow.continuation(self.execute_node.bind(n, result)) for n in next_nodes]

        except Exception as e:
            raise RuntimeError(f"Error executing node '{name}': {e}")

    def run(self, start_node: str, *args, **kwargs) -> Any:
        """
        Start executing the workflow from a given node.
        :param start_node: The entry point of the DAG.
        """
        if start_node not in self._system()._nodes:
            raise ValueError(f"Start node '{start_node}' does not exist in the graph.")
        return workflow.run(self.execute_node.bind(start_node, *args, **kwargs))