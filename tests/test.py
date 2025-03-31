import asyncio
import logging
import os
import tempfile
import subprocess
from enum import Enum
from typing import Dict, Any, Optional, Union, List, TypeVar, Set
from dataclasses import dataclass, field
from pathlib import Path
import importlib.util
import sys
import git
import shutil

from .handlers import ToolHandler, AgentHandler, ValidationError, ExecutionError
from .package_executor import PackageExecutor, CommunicationMode

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class BaseRegistry:
    """
    Registry to manage tools and agents with efficient resource sharing.
    
    The registry is responsible for:
    1. Tracking all registered tools and agents
    2. Cloning and setting up repositories
    3. Managing the PackageExecutor for executing tools and agents
    4. Providing central access to all resources
    """
    base_path: str = field(default_factory=lambda: os.path.expanduser("~/.system_registry"))
    venv_base_path: str = field(default_factory=lambda: os.path.expanduser("~/.system_venvs"))
    max_concurrent_executions: int = 10
    communication_mode: CommunicationMode = CommunicationMode.BATCH
    
    # Private fields
    _tools: Dict[str, ToolHandler] = field(default_factory=dict)
    _agents: Dict[str, AgentHandler] = field(default_factory=dict)
    _cloned_repos: Dict[str, str] = field(default_factory=dict)
    _executor: Optional[PackageExecutor] = None
    _logger: logging.Logger = field(default_factory=lambda: logging.getLogger("BaseRegistry"))
    
    def __post_init__(self) -> None:
        """Initialize the registry and create necessary directories."""
        # Create base directories
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.venv_base_path, exist_ok=True)
        
        # Initialize the PackageExecutor
        self._executor = PackageExecutor(
            communication_mode=self.communication_mode,
            logger=self._logger,
            max_concurrent_executions=self.max_concurrent_executions,
            timeout=300.0,  # 5 minutes default timeout
            buffer_size=2000,  # Larger buffer for complex outputs
            memory_limit_mb=2048.0  # 2GB default memory limit
        )
        
        self._logger.info(f"Registry initialized at {self.base_path}")
        self._logger.info(f"Virtual environments will be stored at {self.venv_base_path}")
    
    def register_tool(
        self, 
        name: str, 
        tool_type: str, 
        **kwargs: Any
    ) -> ToolHandler:
        """
        Register a tool with the registry.
        
        Args:
            name: Unique tool name
            tool_type: Type of tool (Local, Remote)
            **kwargs: Additional tool configuration
            
        Returns:
            ToolHandler instance
            
        Raises:
            ValidationError: If the tool configuration is invalid
        """
        if name in self._tools:
            raise ValidationError(f"Tool '{name}' is already registered")
            
        # Special handling for repo_id
        if 'repo_id' in kwargs:
            repo_id = kwargs.pop('repo_id')
            local_path = self.clone_repo(repo_id)
            
            # Find the entry point
            entry_point = kwargs.pop('entry_point', self._find_entry_point(local_path))
            
            # Create a virtual environment
            venv_path = os.path.join(self.venv_base_path, f"tool_{name}")
            if not os.path.exists(venv_path):
                self._create_venv(venv_path, local_path)
                
            # Update kwargs with local paths
            kwargs.update({
                'dir': local_path,
                'venv_path': venv_path,
                'entry_point': entry_point
            })
            
        # Create and register the tool handler
        tool_handler = ToolHandler(name=name, tool_type=tool_type, **kwargs)
        
        # Inject the executor
        if self._executor:
            tool_handler.set_executor(self._executor)
            
        self._tools[name] = tool_handler
        self._logger.info(f"Tool '{name}' registered successfully")
        
        return tool_handler
    
    def register_agent(
        self, 
        name: str, 
        agent_type: str, 
        **kwargs: Any
    ) -> AgentHandler:
        """
        Register an agent with the registry.
        
        Args:
            name: Unique agent name
            agent_type: Type of agent (Local, Remote)
            **kwargs: Additional agent configuration
            
        Returns:
            AgentHandler instance
            
        Raises:
            ValidationError: If the agent configuration is invalid
        """
        if name in self._agents:
            raise ValidationError(f"Agent '{name}' is already registered")
            
        # Special handling for repo_id
        if 'repo_id' in kwargs:
            repo_id = kwargs.pop('repo_id')
            local_path = self.clone_repo(repo_id)
            
            # Find the entry point
            entry_point = kwargs.pop('entry_point', self._find_entry_point(local_path))
            
            # Create a virtual environment
            venv_path = os.path.join(self.venv_base_path, f"agent_{name}")
            if not os.path.exists(venv_path):
                self._create_venv(venv_path, local_path)
                
            # Update kwargs with local paths
            kwargs.update({
                'dir': local_path,
                'venv_path': venv_path,
                'entry_point': entry_point
            })
            
        # Create and register the agent handler
        agent_handler = AgentHandler(name=name, agent_type=agent_type, **kwargs)
        
        # Inject the executor
        if self._executor:
            agent_handler.set_executor(self._executor)
            
        self._agents[name] = agent_handler
        self._logger.info(f"Agent '{name}' registered successfully")
        
        return agent_handler
    
    def get_tool(self, name: str) -> ToolHandler:
        """
        Get a registered tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            ToolHandler instance
            
        Raises:
            ValidationError: If the tool is not found
        """
        if name not in self._tools:
            raise ValidationError(f"Tool '{name}' not found in registry")
        return self._tools[name]
    
    def get_agent(self, name: str) -> AgentHandler:
        """
        Get a registered agent by name.
        
        Args:
            name: Agent name
            
        Returns:
            AgentHandler instance
            
        Raises:
            ValidationError: If the agent is not found
        """
        if name not in self._agents:
            raise ValidationError(f"Agent '{name}' not found in registry")
        return self._agents[name]
    
    def get_all_tools(self) -> Dict[str, ToolHandler]:
        """Get all registered tools."""
        return self._tools.copy()
    
    def get_all_agents(self) -> Dict[str, AgentHandler]:
        """Get all registered agents."""
        return self._agents.copy()
    
    def clone_repo(self, repo_id: str) -> str:
        """
        Clone a repository and return the local path.
        
        Args:
            repo_id: Repository identifier (URL or path)
            
        Returns:
            Local path to the cloned repository
            
        Raises:
            ValidationError: If cloning fails
        """
        if repo_id in self._cloned_repos:
            # Check if the repo is still valid
            local_path = self._cloned_repos[repo_id]
            if os.path.exists(local_path):
                return local_path
        
        try:
            # Parse repo_id
            if repo_id.startswith("git/"):
                # Format: git/username/repo
                parts = repo_id.split("/")
                if len(parts) < 3:
                    raise ValidationError(f"Invalid git repo format: {repo_id}. Expected: git/username/repo")
                
                username = parts[1]
                repo_name = parts[2]
                url = f"https://github.com/{username}/{repo_name}.git"
            elif repo_id.startswith(("http://", "https://", "git@")):
                # Direct URL
                url = repo_id
                repo_name = url.split("/")[-1].replace(".git", "")
            else:
                raise ValidationError(f"Unsupported repo format: {repo_id}")
            
            # Create a unique directory for the repo
            local_path = os.path.join(self.base_path, "repos", repo_name)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Clone or update the repository
            if os.path.exists(local_path):
                # Update existing repo
                self._logger.info(f"Updating repository at {local_path}")
                repo = git.Repo(local_path)
                origin = repo.remotes.origin
                origin.pull()
            else:
                # Clone new repo
                self._logger.info(f"Cloning repository from {url} to {local_path}")
                git.Repo.clone_from(url, local_path)
            
            # Cache the repo location
            self._cloned_repos[repo_id] = local_path
            return local_path
            
        except git.GitCommandError as e:
            self._logger.error(f"Git error cloning {repo_id}: {str(e)}")
            raise ValidationError(f"Failed to clone repository {repo_id}: {str(e)}") from e
        except Exception as e:
            self._logger.error(f"Error cloning {repo_id}: {str(e)}")
            raise ValidationError(f"Failed to process repository {repo_id}: {str(e)}") from e
    
    def _create_venv(self, venv_path: str, package_path: str) -> None:
        """
        Create a virtual environment and install the package.
        
        Args:
            venv_path: Path to create the virtual environment
            package_path: Path to the package to install
            
        Raises:
            ValidationError: If venv creation or package installation fails
        """
        try:
            # Check if requirements.txt or setup.py exists
            has_requirements = os.path.exists(os.path.join(package_path, "requirements.txt"))
            has_setup = os.path.exists(os.path.join(package_path, "setup.py"))
            
            if not (has_requirements or has_setup):
                self._logger.warning(f"No requirements.txt or setup.py found in {package_path}")
            
            # Create the virtual environment
            self._logger.info(f"Creating virtual environment at {venv_path}")
            subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
            
            # Get the Python executable for the venv
            if os.name == "nt":  # Windows
                python_exe = os.path.join(venv_path, "Scripts", "python.exe")
                pip_exe = os.path.join(venv_path, "Scripts", "pip.exe")
            else:  # Unix-like
                python_exe = os.path.join(venv_path, "bin", "python")
                pip_exe = os.path.join(venv_path, "bin", "pip")
            
            # Upgrade pip
            subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], check=True)
            
            # Install requirements if they exist
            if has_requirements:
                self._logger.info(f"Installing requirements from {package_path}/requirements.txt")
                subprocess.run(
                    [pip_exe, "install", "-r", os.path.join(package_path, "requirements.txt")],
                    check=True
                )
            
            # Install the package if setup.py exists
            if has_setup:
                self._logger.info(f"Installing package from {package_path}")
                subprocess.run(
                    [pip_exe, "install", "-e", package_path],
                    check=True
                )
            
            self._logger.info(f"Virtual environment created and package installed successfully")
            
        except subprocess.CalledProcessError as e:
            self._logger.error(f"Command error creating venv: {str(e)}")
            raise ValidationError(f"Failed to create virtual environment: {str(e)}") from e
        except Exception as e:
            self._logger.error(f"Error creating venv: {str(e)}")
            raise ValidationError(f"Failed to set up virtual environment: {str(e)}") from e
    
    # def _find_entry_point(self, package_path: str) -> str:
    #     """
    #     Find the main entry point script in a package.
        
    #     Args:
    #         package_path: Path to the package
            
    #     Returns:
    #         Relative path to the entry point script
            
    #     Raises:
    #         ValidationError: If no entry point