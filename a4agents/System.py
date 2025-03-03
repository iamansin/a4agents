import ray
from typing import Callable, Dict, Any, Union, Optional, Tuple, List, Literal
from a4agents.System_Utils import SystemNode
import base64
import logging
import networkx as nx
import weakref
from IPython.display import Image, display

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Ray
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

class System:
    """A robust AI agent system using Ray DAGs."""
    
    def __init__(self, name: str):
        self.name = name
        self._nodes = {}  # Stores agent nodes
        self._dag = nx.DiGraph()  
        self._workflow = WorkflowConstructure(self)
        self._DAGExecutor = DAGExecuter(self)
        
    def node(self, func: Optional[Callable] = None, name: Optional[str] = None, resources: Optional[Dict[str, Any]] = None):
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

            if name in self._nodes:
                raise ValueError(f"Node name '{name}' already exists in system '{self.name}'.")

            if any(node.func == func for node in self._nodes.values()):
                raise ValueError(f"Function '{func.__name__}' is already registered under a different name.")

            try:
                node = SystemNode(func, name, resources)
                self._nodes[name] = node
                self._workflow.add_node(node) # Add node to the workflow graph
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

    def nodes_from_dict(self, func_dict: Dict[str, Tuple[Callable, Optional[Dict[str, Any]]]]):
        """
        Adds multiple functions as SystemNodes from a dictionary.
        
        :param func_dict: Dictionary where keys are node names and values are functions.
        :param resources: Optional dictionary specifying resource allocation.
        :raises TypeError: If func_dict is not a dictionary or contains invalid keys/values.
        :raises ValueError: If any node name is already registered.
        """
        if not isinstance(func_dict, dict):
            raise TypeError("Expected a dictionary with {name: function} mapping, but got {type(func_dict).__name__}")

        for name, args in func_dict.items():
            
            try:
                self.node( func= args[0], name = name, resources= args[1])
            except Exception as e:
                logger.error(f"Error adding node '{name}': {str(e)}")
                raise

    def add_route(self, from_node: str, to_node: Optional[str] = None):
        """Adds routes through the system's Router. Handles system-level errors."""
        
        if to_node:
            try:
                self._workflow.add_edges(from_node, to_node)
                logger.info(f"Added direct route from '{from_node}' to '{to_node}'.")
            except Exception as e:
                logger.error(f"Error adding direct route from '{from_node}' to '{to_node}': {str(e)}")
                raise
    
    def add_conditional_route(self, from_node: str, mapping_dict: Dict[str, str]):
        """Adds conditional routes through the system's Router. Handles system-level errors."""
        try:
            self._workflow.conditional_edges(from_node, mapping_dict)
            logger.info(f"Added conditional route from '{from_node}' to '{mapping_dict}'.")
        except Exception as e:
            logger.error(f"Error adding conditional route from '{from_node}' to '{mapping_dict}': {str(e)}")
            raise
        
    def workflow(self, workflow: Dict[str, list]):
        """
        Defines execution order in the system's DAG.

        Parameters:
        - workflow (dict): { "parent_node": ["child_node1", "child_node2", ...] }
        """
        try:
            dag = {}
            for parent, children in workflow.items():
                if parent not in self._nodes:
                    raise ValueError(f"Node '{parent}' not registered in the system.")

                parent_node = self._nodes[parent]

                for child in children:
                    if child not in self._nodes:
                        raise ValueError(f"Node '{child}' not registered in the system.")
                    
                    child_node = self._nodes[child]
                    dag[child] = child_node.bind(parent_node.bind)

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

    def draw_graph(self) -> None:
        """Display the mermaid diagram using mermaid.ink with error handling."""
        
        def generate_mermaid() -> str:
            """Generate the mermaid.js diagram string with proper error handling."""
            if not self._nodes:
                raise RuntimeError("Cannot generate a Mermaid diagram without any nodes.")
            
            graph = self._dag
            mermaid_string = "graph TD\n"

            shape_map = {
                "rectangle": '[\"{label}\"]',
                "diamond": '{{\"{label}\"}}',
                "circle": '(\"{label}\")'
            }
            # !!!!!!!! This code needs proper attentions as self._nodes does not unpack as set of 3..
            for node_name, node_attrs in graph.nodes(data=True):
                shape = node_attrs.get("shape", "circle")  # Get shape, default to circle if not found
                shape_str_mermaid = shape_map[shape].format(label=node_name)
                mermaid_string += f"    {node_name}{shape_str_mermaid}\n"

            # Add edges with labels and conditional styles
            for edge in graph.edges:
                from_node = edge[0]
                to_node = edge[1]
                label = graph[from_node][to_node]["type"]
                edge_style = "-->" if label == "Direct" else "-.->"
                mermaid_string += f"    {from_node} {edge_style}{to_node}\n"

            return mermaid_string.strip()
        try:
            mermaid_string = generate_mermaid()
            base64_string = base64.urlsafe_b64encode(mermaid_string.encode("utf8")).decode("ascii")
            mermaid_url = "https://mermaid.ink/img/" + base64_string
            display(Image(url=mermaid_url))
        except Exception as e:
          raise e
          print(f"Error while generating Mermaid diagram: {e}")
            
            

class WorkflowConstructure:
    def __init__(self, system_reference):
        self._system = weakref.ref(system_reference) 
        self._graph = self._system()._dag 
        self._mapping_dict = {}
        
    def add_node(self, node: SystemNode):
        """
        Adds a node to the workflow graph.

        Args:
            node_name (str): The name of the node.
            node_type (str): The type of the node.
            resources (Optional[Dict[str, Any]]): Resource allocation for the node.

        Raises:
            ValueError: If the node already exists in the graph.
        """
        name = node.name
        sh = "rectangle" if node.is_tool else "circle"
        self._graph.add_node(name, func_ref = node.task, resources= node.resources, shape = sh)
        logger.info(f"Node '{name}' added to the workflow graph.")
        
    def add_edges(self, from_node: str, to_node: str):
        """
        Adds a direct edge between two nodes.
        
        Args:
            from_node (str): Starting node.
            to_node (str): Target node.
        
        Raises:
            ValueError: If the edge already exists or nodes are missing.
        """
        if from_node not in self._system()._nodes:
            raise ValueError(f"Source Node: {from_node} does not found in Graph.")
        
        if to_node not in self._system()._nodes:
            raise ValueError(f"Target Node: {to_node} does not found in Graph.")

        if to_node in self._graph.successors(from_node):
            raise ValueError(f"Route from '{from_node}' to '{to_node}' already exists.")
        
        try:
            self._graph.add_edge(from_node, to_node, type="Direct")
        
        except Exception as e:
            raise
        
    def conditional_edges(self, from_node :str , mapping : Dict[str,str]):
        """
        Adds conditional edges where the transition depends on the output value.
        
        Args:
            from_node (str): The node from which transitions occur.
            mapping (Dict[str, str]): A dictionary mapping outputs to target nodes.
        
        Raises:
            ValueError: If any target node does not exist or the edge already exists.
        """
        if from_node not in self._system()._nodes:
            raise ValueError(f"Source Node: {from_node}' does not found in Graph.")

        existing_successors = set(self._graph.successors(from_node))
  
        for return_value, to_node in mapping.items():
  
            if to_node not in self._graph.nodes:
                raise ValueError(f"Target node '{to_node}' does not exist.")

            if to_node in existing_successors:
                logger.warning(f"Edge from '{from_node}' to '{to_node}' already exists.")
                continue  # Skip adding duplicate edges
            
            try:
                self._graph.add_edge(from_node, to_node, _from = from_node, _to = to_node, type="Conditional", return_value=return_value)
            
            except Exception as e:
                raise
        
        self._mapping_dict[from_node] = mapping