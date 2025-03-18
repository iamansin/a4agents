import os
import json
import shutil
import subprocess
import importlib.util
from typing import Optional, Dict
from pathlib import Path

REGISTRY_FILE = "registry.json"
REGISTRY_DIR = "Registry"
TOOLS_DIR = os.path.join(REGISTRY_DIR, "tools")
AGENTS_DIR = os.path.join(REGISTRY_DIR, "agents")

class ToolHandler:
    def __init__(self, name: str, tool_type: str, endpoint: Optional[str] = None, api_key: Optional[str] = None):
        self.name = name
        self.tool_type = tool_type  # langchain, mcp-server, local, etc.
        self.endpoint = endpoint
        self.api_key = api_key
    
    def to_dict(self):
        return {"name": self.name, "tool_type": self.tool_type, "endpoint": self.endpoint, "api_key": self.api_key}

class AgentHandler:
    def __init__(self, name: str, agent_type: str, repo_url: Optional[str] = None, endpoint: Optional[str] = None):
        self.name = name
        self.agent_type = agent_type  # local, remote (endpoint), github repo
        self.repo_url = repo_url
        self.endpoint = endpoint
    
    def to_dict(self):
        return {"name": self.name, "agent_type": self.agent_type, "repo_url": self.repo_url, "endpoint": self.endpoint}

class Registry:
    def __init__(self):
        os.makedirs(TOOLS_DIR, exist_ok=True)
        os.makedirs(AGENTS_DIR, exist_ok=True)
        if not os.path.exists(REGISTRY_FILE):
            with open(REGISTRY_FILE, "w") as f:
                json.dump({"tools": {}, "agents": {}}, f)
        self._load_registry()

    def _load_registry(self):
        with open(REGISTRY_FILE, "r") as f:
            self.registry = json.load(f)
    
    def _save_registry(self):
        with open(REGISTRY_FILE, "w") as f:
            json.dump(self.registry, f, indent=4)

    def add_tool(self, name: str, tool_type: str, endpoint: Optional[str] = None, api_key: Optional[str] = None):
        if name in self.registry["tools"]:
            raise ValueError(f"Tool '{name}' is already registered.")
        
        tool = ToolObject(name, tool_type, endpoint, api_key)
        self.registry["tools"][name] = tool.to_dict()
        self._save_registry()

    def add_agent(self, name: str, agent_type: str, repo_url: Optional[str] = None, endpoint: Optional[str] = None):
        if name in self.registry["agents"]:
            raise ValueError(f"Agent '{name}' is already registered.")
        
        agent = AgentObject(name, agent_type, repo_url, endpoint)
        self.registry["agents"][name] = agent.to_dict()
        self._save_registry()

        if repo_url:
            self._clone_repo(repo_url, os.path.join(AGENTS_DIR, name))

    def _clone_repo(self, repo_url: str, destination: str):
        if os.path.exists(destination):
            shutil.rmtree(destination)
        subprocess.run(["git", "clone", repo_url, destination], check=True)

    def get_tool(self, name: str) -> Dict:
        return self.registry["tools"].get(name, None)

    def get_agent(self, name: str) -> Dict:
        return self.registry["agents"].get(name, None)
    
    def remove_tool(self, name: str):
        if name not in self.registry["tools"]:
            raise ValueError(f"Tool '{name}' not found.")
        del self.registry["tools"][name]
        self._save_registry()

    def remove_agent(self, name: str):
        if name not in self.registry["agents"]:
            raise ValueError(f"Agent '{name}' not found.")
        
        agent_dir = os.path.join(AGENTS_DIR, name)
        if os.path.exists(agent_dir):
            shutil.rmtree(agent_dir)
        
        del self.registry["agents"][name]
        self._save_registry()

    def load_tool(self, name: str):
        tool_info = self.get_tool(name)
        if not tool_info:
            raise ValueError(f"Tool '{name}' not found in registry.")
        
        if tool_info["tool_type"] == "local":
            module_path = os.path.join(TOOLS_DIR, name, "tool.py")
            return self._load_module(module_path)
        return tool_info

    def load_agent(self, name: str):
        agent_info = self.get_agent(name)
        if not agent_info:
            raise ValueError(f"Agent '{name}' not found in registry.")
        
        if agent_info["agent_type"] == "local":
            module_path = os.path.join(AGENTS_DIR, name, "agent.py")
            return self._load_module(module_path)
        return agent_info

    def _load_module(self, module_path: str):
        if not os.path.exists(module_path):
            raise ValueError(f"Module file '{module_path}' not found.")
        
        module_name = Path(module_path).stem
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

# Example Usage
if __name__ == "__main__":
    reg = Registry()
    reg.add_tool("websearch", "langchain", endpoint="https://api.example.com", api_key="secret")
    reg.add_agent("deep-research", "github", repo_url="https://github.com/user/deep-research-agent")
    
    print(reg.get_tool("websearch"))
    print(reg.get_agent("deep-research"))
