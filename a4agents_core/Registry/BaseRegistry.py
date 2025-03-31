#Here we have to create our own Error class to manage the errors in the registry.
#The Errors can be:
#1. NO TRANSPORT file found
#2. No command found in the configuration file
#3. There was some problem while removing the agents/tools.

import os
import json
import shutil
import asyncio
import logging
import importlib.util
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, asdict
import aiohttp
from urllib.parse import urlparse

from .WheelInstaller import WheelInstaller
from Utils._Singleton import Singleton
from .Handlers import ToolHandler, AgentHandler, ETYPES, ToolHandlerError
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Registry")

# Constants

    
class ValidationError(Exception):
    """Custom exception for validation errors in the registry."""
    pass

class ToolCreationError(Exception):
    pass

# Here we are expecting the Registry to act as a Singelton type of object, 
# but we are not enforcing the strictness of the Singleton pattern as,
# this class is not an object created by the user but is created by the system itself.
# if in rare case another instance of this class will be created then there will not Singleton Error raised.
# The other class will also refer to this one only.
@Singleton(strict=False,thread_safe=True,debug=False)
class Registry:
    """
    A registry for managing tools and agents.
    
    This class provides functionality to add, remove, and load tools and agents.
    It also provides methods to check for the existence of tools and agents in
    remote repositories or endpoints.
    """
    REGISTRY_FILE = "registry.json"
    REGISTRY_DIR = "Registry"
    TOOLS_DIR = os.path.join(REGISTRY_DIR, "Tools")
    AGENTS_DIR = os.path.join(REGISTRY_DIR, "Agents")
    BASE_VENV_DIR = os.path.join(REGISTRY_DIR, "venv")
    os.makedirs(TOOLS_DIR, exist_ok=True)
    os.makedirs(AGENTS_DIR, exist_ok=True)
        
        # Initialize registry file if it doesn't exist
    if not os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "w") as f:
                json.dump({"tools": {}, "agents": {}}, f)
        
    _lock = asyncio.Lock()  # Lock for thread-safe operations
    wheel_installer = WheelInstaller(base_venv_dir=BASE_VENV_DIR)
    
    def __init__(self):
        self._load_registry()
        
    def _load_registry(self) -> None:
        """Load the registry from the registry file."""
        try:
            with open(REGISTRY_FILE, "r") as f:
                self.registry = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error loading registry file: {REGISTRY_FILE}. Creating new registry.")
            self.registry = {"tools": {}, "agents": {}}
            self._save_registry()
    
    def _save_registry(self) -> None:
        """Save the registry to the registry file."""
        try:
            with open(REGISTRY_FILE, "w") as f:
                json.dump(self.registry, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
            raise IOError(f"Failed to save registry: {e}")

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        Get a list of all tools in the registry.
        
        Returns:
            List[Dict[str, Any]]: List of tool information
        """
        return list(self.registry["tools"].values())

    def get_all_agents(self) -> List[Dict[str, Any]]:
        """
        Get a list of all agents in the registry.
        
        Returns:
            List[Dict[str, Any]]: List of agent information
        """
        return list(self.registry["agents"].values())
    
    def get_tool_names(self) -> List[str]:
        """
        Get a list of all tool names in the registry.
        """
        return list(self.registry["tools"].keys())
    
    def get_agent_names(self) -> List[str]:
        """
        Get a list of all tool names in the registry.
        """
        return list(self.registry["tools"].keys())

    def get_registry_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the registry.
        
        Returns:
            Dict[str, Any]: Summary of the registry
        """
        return {
            "total_tools": len(self.registry["tools"]),
            "total_agents": len(self.registry["agents"]),
            "tools": list(self.registry["tools"].keys()),
            "agents": list(self.registry["agents"].keys())
        }
        
    #here we have to use lazy downloading and creation of the handlers at the time of compilation
    @classmethod
    async def add_tool(cls, 
                    name: str, 
                    tool_type: str, 
                    func : Optional[callable] = None,
                    endpoint: Optional[str] = None, 
                    api_key: Optional[str] = None,
                    repo_url: Optional[str] = None) -> ToolHandler:
        
        
        if tool_type == ETYPES.REMOTE:
            try:
                await cls._validate_endpoint(endpoint, api_key)

            except Exception as e:
                raise ToolHandlerError(f"There was some error while validating endpoint and api-key : {str(e)}") from e
            
        elif tool_type == ETYPES.LOCAL:
            try:
                parsed_url = urlparse(repo_url)
                path_parts = parsed_url.path.strip('/').split('/')
                _owner, repo = path_parts[0], path_parts[1]
                repo_dir = os.path.join(cls.TOOLS_DIR, repo)
                # Clone the repository only after confirming YAML exists
                package_path = await cls._clone_repo_async(repo_url, repo_dir)
                venv_path, executable_path = await cls.wheel_installer.install_wheel(package_path)
                
            except Exception as e:
                # Clean up resources on error
                if os.path.exists(repo_dir):
                    shutil.rmtree(repo_dir)
                raise ToolHandlerError(f"There was some problem while creating Local tool : {str(e)}") from e
            
        
        # Create the ToolHandler instance
        toolhandler = ToolHandler(
                    name=name,
                    tool_type = tool_type,
                    func=func,
                    endpoint=endpoint,
                    api_key=api_key,
                    venv_path = venv_path,
                    entry_point = executable_path,
                    dir = repo_dir
                )
                
                # Only lock when updating the registry
        async with cls._lock:
                    # Double-check the tool doesn't exist (in case of race condition)
            if name in cls.registry["tools"]:
                raise ValueError(f"Tool '{name}' is already registered.")
                    
            cls.registry["tools"][name] = toolhandler.to_dict()
            cls._save_registry()
                
        return toolhandler
    
    async def add_agent(self, 
                    name: str, 
                    agent_type: str, 
                    endpoint: Optional[str] = None, 
                    api_key: Optional[str] = None,
                    repo_url: Optional[str] = None) -> AgentHandler:
        """
        Add an agent to the registry.
        
        Args:
            name: Name of the agent
            agent_type: Type of the agent (LOCAL, REMOTE, GITHUB)
            repo_url: URL to the GitHub repository
            endpoint: API endpoint for remote agents
            command: Command to execute the agent
            
        Returns:
            AgentHandler: The created agent handler
            
        Raises:
            ValueError: If an agent with the same name already exists
            ValidationError: If validation fails
        """
        # Check if agent already exists (quick non-locked check)
        if not endpoint and not repo_url:
            raise ValidationError("Agent must have either an endpoint or repository URL")
        
        if name in self.registry["agents"]:
            raise ValueError(f"Agent '{name}' is already registered.")
        
        try:
            # Validate inputs based on agent type before acquiring any locks
            if agent_type == ETYPES.MCP_REMOTE:
                if not endpoint:
                    raise ValidationError("MCP agents must have an endpoint")
                
                # Validate the endpoint URL format and accessibility
                await self._validate_endpoint(endpoint)
                
            elif agent_type == ETYPES.MCP_LOCAL:
                # Extract repo info (owner, repo) from URL
                parsed_url = urlparse(repo_url)
                path_parts = parsed_url.path.strip('/').split('/')
                owner, repo = path_parts[0], path_parts[1]
                repo_dir = os.path.join(AGENTS_DIR, repo)
                print("The repo is ")
                package_path = await self._clone_repo_async(repo_url, repo_dir)
                venv_path, executable_path= await self.wheel_installer.install_wheel(package_path)
                
            else:
                raise ValidationError(f"Unsupported agent type: {agent_type}")

            # Create the agent handler
            agent = AgentHandler(
                name=name,
                agent_type=agent_type,
                repo_url=repo_url,
                endpoint=endpoint,
                venv_path = venv_path,
                entry_popint = executable_path,
                dir = repo_dir
            )
            
            # Inject the executor
            if self._executor:
                agent.set_executor(self._executor)
            
            # Only lock when updating the registry
            async with self._lock:
                # Double-check the agent doesn't exist (in case of race condition)
                if name in self.registry["agents"]:
                    raise ValueError(f"Agent '{name}' is already registered.")
                
                self.registry["agents"][name] = agent.to_dict()
                self._save_registry()
            
            return agent
            
        except Exception as e:
            # Clean up resources on error
            if os.path.exists(repo_dir):
                shutil.rmtree(repo_dir)
            
            if isinstance(e, ValidationError):
                raise
            else:
                raise ValidationError(f"Failed to add agent '{name}': {str(e)}")
            
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a tool from the registry.
        
        Args:
            name: Name of the tool
            
        Returns:
            Optional[Dict[str, Any]]: Tool information, or None if not found
        """
        return self.registry["tools"].get(name,None)

    def get_agent(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get an agent from the registry.
        
        Args:
            name: Name of the agent
            
        Returns:
            Optional[Dict[str, Any]]: Agent information, or None if not found
        """
        return self.registry["agents"].get(name,None)

    async def remove_tool(self, name: str) -> bool:
        """
        Remove a tool from the registry.

        Args:
            name (str): Name of the tool to remove.

        Returns:
            bool: True if the tool was removed, False otherwise.

        Raises:
            ValueError: If the tool is not found.
        """

        # Check existence outside the lock
        tool_handler :ToolHandler = self.registry["tools"].get(name, None)
        if not tool_handler:
            raise ValueError(f"Tool '{name}' not found.")

        tool_dir = tool_handler.dir
        # Remove tool directory asynchronously
        try:
            if tool_dir and os.path.exists(tool_dir):
                await asyncio.to_thread(shutil.rmtree, tool_dir, ignore_errors=True)
        

            # Safely remove from registry
            async with self._lock:
                if name in self.registry["tools"]:  # Double-check inside lock to avoid race condition
                    del self.registry["tools"][name]
                    self._save_registry()
        
        except Exception as e:
            raise e

        return True

    async def remove_agent(self, name: str) -> bool:
        """
        Remove an agent from the registry.

        Args:
            name (str): Name of the agent to remove.

        Returns:
            bool: True if the agent was removed, False otherwise.

        Raises:
            ValueError: If the agent is not found.
        """

        # Check existence outside the lock
        agent_handler :AgentHandler = self.registry["agents"].get(name)
        if not agent_handler:
            raise ValueError(f"Agent '{name}' not found.")

        agent_dir  = agent_handler.dir
        
        # Remove agent directory asynchronously
        try:
            if agent_dir and os.path.exists(agent_dir):
                await asyncio.to_thread(shutil.rmtree, agent_dir, ignore_errors=True)

            # Safely remove from registry
            async with self._lock:
                if name in self.registry["agents"]:  # Double-check inside lock to avoid race condition
                    del self.registry["agents"][name]
                    self._save_registry()

        except Exception as e:
            raise e

        return True
    
    async def _validate_endpoint(self, endpoint: str, api_key: Optional[str] = None, TEST :bool = False) -> None:
        """
        Validate a remote tool endpoint by checking its format and accessibility.
        
        Args:
            endpoint: The URL endpoint to validate
            api_key: Optional API key for authentication
            
        Raises:
            ValidationError: If the endpoint is invalid or inaccessible
        """
        # Validate URL format
        try:
            parsed_url = urlparse(endpoint)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                raise ValidationError(f"Invalid endpoint URL format: {endpoint}")
        except Exception:
            raise ValidationError(f"Invalid endpoint URL: {endpoint}")
        
        if TEST:
            # Check endpoint accessibility with timeout
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(endpoint, headers=headers, timeout=5) as response:
                        if response.status >= 400:
                            raise ValidationError(
                                f"Endpoint returned error status {response.status}: {endpoint}"
                            )
            except aiohttp.ClientError as e:
                raise ValidationError(f"Failed to connect to endpoint {endpoint}: {str(e)}")
            except asyncio.TimeoutError:
                raise ValidationError(f"Connection to endpoint {endpoint} timed out")

    async def _clone_repo_async(self, repo_url: str, target_dir: str) -> None:
        """
        Clone a git repository asynchronously.
        
        Args:
            repo_url: URL of the git repository
            target_dir: Directory to clone into
        
        Raises:
            ValidationError: If cloning fails
        """
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        
        os.makedirs(target_dir, exist_ok=True)
        
        process = await asyncio.create_subprocess_exec(
            'git', 'clone', '--depth=1', repo_url, target_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        _ , stderr = await process.communicate()
        
        if process.returncode != 0:
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            raise ValidationError(f"Failed to clone repository: {stderr.decode().strip()}")
        
        whl_files = list(Path(target_dir).rglob("*.whl"))

        if not whl_files:
            raise ValidationError("No .whl file found in the cloned repository.")

        return str(whl_files[0])
   
    async def remove_tools(self, names: list[str]) -> dict[str, bool]:
        """
        Remove multiple tools asynchronously with optimized execution.

        Args:
            names (list[str]): List of tool names to remove.

        Returns:
            dict[str, bool]: A dictionary mapping tool names to True (removed) or False (failed).
        """

        tasks = [self.remove_tool(name) for name in names]  
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {name: (res is True) for name, res in zip(names, results)}

    async def remove_agents(self, names: list[str]) -> dict[str, bool]:
        """
        Remove multiple agents asynchronously with optimized execution.

        Args:
            names (list[str]): List of agent names to remove.

        Returns:
            dict[str, bool]: A dictionary mapping agent names to True (removed) or False (failed).
        """

        tasks = [self.remove_agent(name) for name in names]  
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {name: (res is True) for name, res in zip(names, results)}

    async def load_tool(self, name: str) -> Union[Any, Dict[str, Any]]:
        """
        Load a tool from the registry.
        
        This method loads a tool and returns either a module (for local tools) or
        the tool information (for remote tools).
        
        Args:
            name: Name of the tool
            
        Returns:
            Union[Any, Dict[str, Any]]: Loaded tool module or tool information
            
        Raises:
            ValueError: If the tool is not found or cannot be loaded
        """
        tool_handler : ToolHandler = self.get_tool(name) #Here tool_info is of type ToolHandler class object.
        if not tool_handler:
            raise ValueError(f"Tool '{name}' not found in registry.")
        
        if tool_handler.tool_type == ETYPES.MCP_LOCAL:
            # First check if we need to use the command
            if tool_handler.command:
                return tool_handler
            
            # If no command or config, try to load the module
            module_path = os.path.join(TOOLS_DIR, name, "tool.py")
            try:
                return await self._load_module_async(module_path)
            except Exception as e:
                raise ValueError(f"Failed to load tool module: {e}")
        
        return tool_info

    async def load_agent(self, name: str) -> Union[Any, Dict[str, Any]]:
        """
        Load an agent from the registry.
        
        This method loads an agent and returns either a module (for local agents) or
        the agent information (for remote agents).
        
        Args:
            name: Name of the agent
            
        Returns:
            Union[Any, Dict[str, Any]]: Loaded agent module or agent information
            
        Raises:
            ValueError: If the agent is not found or cannot be loaded
        """
        agent_handler : AgentHandler = await self.get_agent(name)
        if not agent_handler:
            raise ValueError(f"Agent '{name}' not found in registry.")
        
        if agent_handler.agent_type == ETYPES.MCP_LOCAL:
            # First check if we need to use the command
            if agent_handler.command:
                return agent_handler
            
            # If no command or config, try to load the module
            module_path = os.path.join(AGENTS_DIR, name, "agent.py")
            try:
                return await self._load_module_async(module_path)
            except Exception as e:
                raise ValueError(f"Failed to load agent module: {e}")
        
        return agent_handler
    
#------------------------------------------------------------------------------->
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # async def add_tool(self, 
    #                 name: str, 
    #                 tool_type: str, 
    #                 endpoint: Optional[str] = None, 
    #                 api_key: Optional[str] = None,
    #                 command: Optional[str] = None,
    #                 repo_url: Optional[str] = None) -> ToolHandler:
    #     """
    #     Validate and add a tool to the registry in one step.

    #     For remote tools, an endpoint is required. For local tools, either a 
    #     repository URL or a command is required. If a repository URL is provided 
    #     and no command is given, the repository is checked for a YAML 
    #     configuration file without cloning the entire repo first.

    #     Args:
    #         name: Name of the tool.
    #         tool_type: Type of the tool ("remote" or ETYPES.MCP_LOCAL).
    #         endpoint: Endpoint for remote tools.
    #         api_key: API key for authentication (remote tools).
    #         command: Command to run the tool.
    #         repo_url: Repository URL for local tools.

    #     Returns:
    #         ToolHandler: The created and validated ToolHandler instance.

    #     Raises:
    #         ValueError: If the tool is already registered.
    #         ValidationError: If any validation check fails.
    #     """
    #     # Check if tool already exists (quick non-locked check)
        
    #     if not endpoint and not repo_url:
    #         raise ValidationError("Tool must have either an endpoint, repository URL")
        
    #     if name in self.registry["tools"]:
    #         raise ValueError(f"Tool '{name}' is already registered.")
        
    #     config_path = None
    #     # temp_dir = None
    #     try:
    #         # Validate inputs based on tool type before acquiring any locks
    #         if tool_type == ETYPES.MCP_REMOTE:
    #             if not endpoint:
    #                 raise ValidationError("MCP tools must have an endpoint")
                
    #             # Validate the endpoint URL format and accessibility
    #             await self._validate_endpoint(endpoint, api_key)
                
    #         elif tool_type == ETYPES.MCP_LOCAL:
    #                 # Extract repo info (owner, repo) from URL
    #             parsed_url = urlparse(repo_url)
    #             path_parts = parsed_url.path.strip('/').split('/')
    #             owner, repo = path_parts[0], path_parts[1]
    #             repo_dir = os.path.join(TOOLS_DIR, repo)
                
    #             if repo_url and not command:
    #                 # Check for YAML file existence without cloning the entire repository
    #                 yaml_exists, _ = await self._check_yaml_file_exists( owner, repo)
                    
    #                 if not yaml_exists:
    #                     raise ValidationError(
    #                         f"No YAML configuration file found for tool '{name}' and no command provided"
    #                     )
                    
    #                 # Clone the repository only after confirming YAML exists
    #                 await self._clone_repo_async(repo_url, repo_dir)
                    
    #                 # Find the config file in the cloned repository
    #                 config_path = await self._find_config_file(repo_dir)
    #                 if not config_path:
    #                     raise ValidationError(f"No TRANSPORT configuration file found for tool '{repo}'")
    #                 try:
    #                     command = await self._extract_command(config_path)
    #                 except ValueError as e:
    #                     raise ValidationError(str(e))
    #         else:
    #             raise ValidationError(f"Unsupported tool type: {tool_type}")

    #         # Create the ToolHandler instance
    #         tool = ToolHandler(
    #             name=name,
    #             tool_type=tool_type,
    #             endpoint=endpoint,
    #             api_key=api_key,
    #             command=command,
    #             config_path=config_path,
    #             dir = repo_dir
    #         )
            
    #         # Only lock when updating the registry
    #         async with self._lock:
    #             # Double-check the tool doesn't exist (in case of race condition)
    #             if name in self.registry["tools"]:
    #                 raise ValueError(f"Tool '{name}' is already registered.")
                
    #             self.registry["tools"][name] = tool.to_dict()
    #             self._save_registry()
            
    #         return tool
            
    #     except Exception as e:
    #         # Clean up resources on error
    #         if os.path.exists(repo_dir):
    #             shutil.rmtree(repo_dir)
    #         # if temp_dir and os.path.exists(temp_dir):
    #         #     shutil.rmtree(temp_dir)
            
    #         if isinstance(e, ValidationError):
    #             raise
    #         else:
    #             raise ValidationError(f"Failed to add tool '{name}': {str(e)}")

    # async def add_agent(self,
    #                 name: str,
    #                 agent_type: str,
    #                 repo_url: Optional[str] = None,
    #                 endpoint: Optional[str] = None,
    #                 command: Optional[str] = None) -> AgentHandler:
    #     """
    #     Add an agent to the registry.
        
    #     Args:
    #         name: Name of the agent
    #         agent_type: Type of the agent (LOCAL, REMOTE, GITHUB)
    #         repo_url: URL to the GitHub repository
    #         endpoint: API endpoint for remote agents
    #         command: Command to execute the agent
            
    #     Returns:
    #         AgentHandler: The created agent handler
            
    #     Raises:
    #         ValueError: If an agent with the same name already exists
    #         ValidationError: If validation fails
    #     """
    #     # Check if agent already exists (quick non-locked check)
    #     if not endpoint and not repo_url:
    #         raise ValidationError("Agent must have either an endpoint or repository URL")
        
    #     if name in self.registry["agents"]:
    #         raise ValueError(f"Agent '{name}' is already registered.")
        
    #     config_path = None
        
    #     try:
    #         # Validate inputs based on agent type before acquiring any locks
    #         if agent_type == "REMOTE":
    #             if not endpoint:
    #                 raise ValidationError("REMOTE agents must have an endpoint")
                
    #             # Validate the endpoint URL format and accessibility
    #             await self._validate_endpoint(endpoint)
                
    #         elif agent_type == ETYPES.MCP_LOCAL:
    #             # Extract repo info (owner, repo) from URL
    #             parsed_url = urlparse(repo_url)
    #             path_parts = parsed_url.path.strip('/').split('/')
    #             owner, repo = path_parts[0], path_parts[1]
    #             repo_dir = os.path.join(AGENTS_DIR, repo)
                
    #             if repo_url and not command:
    #                 # Check for YAML file existence without cloning the entire repository
    #                 yaml_exists, _ = await self._check_yaml_file_exists(owner, repo)
                    
    #                 if not yaml_exists:
    #                     raise ValidationError(
    #                         f"No YAML configuration file found for agent '{name}' and no command provided"
    #                     )
                    
    #                 # Clone the repository only after confirming YAML exists
    #                 await self._clone_repo_async(repo_url, repo_dir)
                    
    #                 # Find the config file in the cloned repository
    #                 config_path = await self._find_config_file(repo_dir)
    #                 if not config_path:
    #                     raise ValidationError(f"No TRANSPORT configuration file found for agent '{repo}'")
                    
    #                 try:
    #                     command = await self._extract_command(config_path)
    #                 except ValueError as e:
    #                     raise ValidationError(str(e))
                
    #         else:
    #             raise ValidationError(f"Unsupported agent type: {agent_type}")

    #         # Create the agent handler
    #         agent = AgentHandler(
    #             name=name,
    #             agent_type=agent_type,
    #             repo_url=repo_url,
    #             endpoint=endpoint,
    #             command=command,
    #             config_path=config_path
    #         )
            
    #         # Only lock when updating the registry
    #         async with self._lock:
    #             # Double-check the agent doesn't exist (in case of race condition)
    #             if name in self.registry["agents"]:
    #                 raise ValueError(f"Agent '{name}' is already registered.")
                
    #             self.registry["agents"][name] = agent.to_dict()
    #             self._save_registry()
            
    #         return agent
            
    #     except Exception as e:
    #         # Clean up resources on error
    #         if os.path.exists(repo_dir):
    #             shutil.rmtree(repo_dir)
            
    #         if isinstance(e, ValidationError):
    #             raise
    #         else:
    #             raise ValidationError(f"Failed to add agent '{name}': {str(e)}")

    # async def _check_yaml_file_exists(self, owner: str, repo: str) -> Tuple[bool, Optional[str]]:
    #     """
    #     Check if a YAML configuration file exists in the repository without cloning the entire repo.
        
    #     Args:
    #         repo_url: URL of the git repository
    #         tool_name: Name of the tool
            
    #     Returns:
    #         Tuple[bool, Optional[str]]: Whether the YAML file exists and its path if found
    #     """

    #     # Check for common YAML file pattern
            
    #     async with aiohttp.ClientSession() as session:
    #         for yaml_path in CONFIG_FILE_TYPES:
    #             api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{yaml_path}"
                    
    #             try:
    #                 async with session.get(api_url) as response:
    #                     if response.status == 200:
    #                         return True, yaml_path
    #             except Exception:
    #                     continue
            
    #     return False, None
        
        # if len(path_parts) < 2 or 'github.com' not in parsed_url.netloc:
        #     # For non-GitHub repos or other formats, we need to clone
        #     # Create a temporary directory for a sparse checkout
        #     temp_dir = tempfile.mkdtemp(prefix=f"tool-check-{tool_name}-")
            
        #     try:
        #         # Initialize git repo
        #         process = await asyncio.create_subprocess_exec(
        #             'git', 'init',
        #             cwd=temp_dir,
        #             stdout=asyncio.subprocess.PIPE,
        #             stderr=asyncio.subprocess.PIPE
        #         )
        #         _, _ = await process.communicate()
                
        #         # Add remote
        #         process = await asyncio.create_subprocess_exec(
        #             'git', 'remote', 'add', 'origin', repo_url,
        #             cwd=temp_dir,
        #             stdout=asyncio.subprocess.PIPE,
        #             stderr=asyncio.subprocess.PIPE
        #         )
        #         _, _ = await process.communicate()
                
        #         # Configure sparse checkout
        #         process = await asyncio.create_subprocess_exec(
        #             'git', 'config', 'core.sparseCheckout', 'true',
        #             cwd=temp_dir,
        #             stdout=asyncio.subprocess.PIPE,
        #             stderr=asyncio.subprocess.PIPE
        #         )
        #         _, _ = await process.communicate()
                
        #         # Create sparse-checkout file with YAML patterns
        #         sparse_file = os.path.join(temp_dir, '.git', 'info', 'sparse-checkout')
        #         os.makedirs(os.path.dirname(sparse_file), exist_ok=True)
        #         with open(sparse_file, 'w') as f:
        #             f.write(f"{tool_name}.yml\n{tool_name}.yaml\n*.yml\n*.yaml\n")
                
        #         # Fetch only the needed files
        #         process = await asyncio.create_subprocess_exec(
        #             'git', 'fetch', '--depth=1', 'origin', 'main',
        #             cwd=temp_dir,
        #             stdout=asyncio.subprocess.PIPE,
        #             stderr=asyncio.subprocess.PIPE
        #         )
        #         stdout, stderr = await process.communicate()
                
        #         if process.returncode != 0:
        #             # Try 'master' branch if 'main' fails
        #             process = await asyncio.create_subprocess_exec(
        #                 'git', 'fetch', '--depth=1', 'origin', 'master',
        #                 cwd=temp_dir,
        #                 stdout=asyncio.subprocess.PIPE,
        #                 stderr=asyncio.subprocess.PIPE
        #             )
        #             stdout, stderr = await process.communicate()
                
        #         # Checkout the files
        #         process = await asyncio.create_subprocess_exec(
        #             'git', 'checkout', 'FETCH_HEAD',
        #             cwd=temp_dir,
        #             stdout=asyncio.subprocess.PIPE,
        #             stderr=asyncio.subprocess.PIPE
        #         )
        #         _, _ = await process.communicate()
                
        #         # Check for YAML files
        #         yaml_files = glob.glob(os.path.join(temp_dir, "*.y*ml")) + \
        #                     glob.glob(os.path.join(temp_dir, f"{tool_name}.y*ml"))
                
        #         if yaml_files:
        #             return True, yaml_files[0]
        #         return False, None
                
        #     finally:
        #         # Clean up the temporary directory
        #         if os.path.exists(temp_dir):
        #             shutil.rmtree(temp_dir)
        
    #     # For GitHub repos, we can use the GitHub API to check file existence

    # async def _find_config_file(self, repo_dir: str) -> Optional[str]:
    #     """
    #     Find a YAML configuration file in the repository.
        
    #     Args:
    #         tool_name: Name of the tool
    #         repo_dir: Path to the repository directory
        
    #     Returns:
    #         Optional[str]: Path to the configuration file if found, None otherwise
    #     """
    #     # Look for tool-specific YAML files first
    #     config_files = [
    #         os.path.join(repo_dir, f"TRANSPORT.yml"),
    #         os.path.join(repo_dir, f"TRANSPORT.yaml")
    #     ]
        
    #     # Add general YAML files as fallback
    #     config_files.extend(glob.glob(os.path.join(repo_dir, "*.yml")))
    #     config_files.extend(glob.glob(os.path.join(repo_dir, "*.yaml")))
        
    #     for config_file in config_files:
    #         if os.path.isfile(config_file):
    #             return config_file
        
    #     return None

# WE have to edit this function such that we are considering two file type, YAML/YML, 
# also the structure of the file so that we can extract commands from therw
# here we also have to apply logic for hanlding the Agents as well as Tools command extraction,
# as there will be states and other arguments required for running the agent.

    # async def _extract_command(self, config_path: str) -> str:
    #     """
    #     Extract the command from a YAML configuration file.
        
    #     Args:
    #         config_path: Path to the configuration file
            
    #     Returns:
    #         str: Extracted command
            
    #     Raises:
    #         ValueError: If no command is found in the configuration file
    #     """
    #     try:
    #         with open(config_path, 'r') as file:
    #             config = yaml.safe_load(file)
            
    #         command = config.get('command')
    #         if not command:
    #             raise ValueError(f"No command found in configuration file: {config_path}")
            
    #         return command
    #     except Exception as e:
    #         logger.error(f"Error extracting command from {config_path}: {e}")
            # raise ValueError(f"Failed to extract command from configuration file: {e}")

    # async def _load_module_async(self, module_path: str) -> Any:
    #     """
    #     Load a Python module asynchronously.
        
    #     Args:
    #         module_path: Path to the Python module
            
    #     Returns:
    #         Any: Loaded module
            
    #     Raises:
    #         ValueError: If the module file is not found or cannot be loaded
    #     """
    #     if not os.path.exists(module_path):
    #         raise ValueError(f"Module file '{module_path}' not found.")
        
    #     try:
    #         # Run in a thread to avoid blocking the event loop
    #         loop = asyncio.get_event_loop()
    #         return await loop.run_in_executor(None, self._load_module_sync, module_path)
    #     except Exception as e:
    #         logger.error(f"Error loading module '{module_path}': {e}")
    #         raise ValueError(f"Failed to load module: {e}")

    # def _load_module_sync(self, module_path: str) -> Any:
    #     """
    #     Load a Python module synchronously.
        
    #     Args:
    #         module_path: Path to the Python module
            
    #     Returns:
    #         Any: Loaded module
    #     """
    #     module_name = Path(module_path).stem
    #     spec = importlib.util.spec_from_file_location(module_name, module_path)
    #     module = importlib.util.module_from_spec(spec)
    #     spec.loader.exec_module(module)
    #     return module

