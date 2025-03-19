import os
import json
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from BaseRegistry import Registry, ToolHandler, AgentHandler, ValidationError

# Constants for testing
TEST_REGISTRY_FILE = "test_registry.json"
TEST_TOOLS_DIR = "test_tools"
TEST_AGENTS_DIR = "test_agents"

@pytest.fixture
def setup_registry():
    """Fixture to set up a test Registry instance."""
    # Patch constants to use test paths
    with patch("d:\a4agents\a4agents-Lib\Registry\BaseRegistry.REGISTRY_FILE", TEST_REGISTRY_FILE), \
         patch("d:\a4agents\a4agents-Lib\Registry\BaseRegistry.TOOLS_DIR", TEST_TOOLS_DIR), \
         patch("d:\a4agents\a4agents-Lib\Registry\BaseRegistry.AGENTS_DIR", TEST_AGENTS_DIR):
        # Create test directories
        os.makedirs(TEST_TOOLS_DIR, exist_ok=True)
        os.makedirs(TEST_AGENTS_DIR, exist_ok=True)
        yield Registry()
        # Cleanup test files and directories
        if os.path.exists(TEST_REGISTRY_FILE):
            os.remove(TEST_REGISTRY_FILE)
        if os.path.exists(TEST_TOOLS_DIR):
            shutil.rmtree(TEST_TOOLS_DIR)
        if os.path.exists(TEST_AGENTS_DIR):
            shutil.rmtree(TEST_AGENTS_DIR)

@pytest.mark.asyncio
async def test_add_tool(setup_registry):
    """Test adding a tool to the registry."""
    registry = setup_registry
    tool_name = "test_tool"
    tool_type = "LOCAL"
    command = "echo 'Hello, World!'"
    
    tool = await registry.add_tool(name=tool_name, tool_type=tool_type, command=command)
    assert tool.name == tool_name
    assert tool.tool_type == tool_type
    assert tool.command == command
    assert tool_name in registry.registry["tools"]

@pytest.mark.asyncio
async def test_add_tool_validation_error(setup_registry):
    """Test adding a tool with invalid data."""
    registry = setup_registry
    with pytest.raises(ValidationError):
        await registry.add_tool(name="", tool_type="LOCAL")

@pytest.mark.asyncio
async def test_remove_tool(setup_registry):
    """Test removing a tool from the registry."""
    registry = setup_registry
    tool_name = "test_tool"
    tool_type = "LOCAL"
    command = "echo 'Hello, World!'"
    
    await registry.add_tool(name=tool_name, tool_type=tool_type, command=command)
    assert tool_name in registry.registry["tools"]
    
    result = await registry.remove_tool(tool_name)
    assert result is True
    assert tool_name not in registry.registry["tools"]

@pytest.mark.asyncio
async def test_add_agent(setup_registry):
    """Test adding an agent to the registry."""
    registry = setup_registry
    agent_name = "test_agent"
    agent_type = "local"
    command = "python agent.py"
    
    agent = await registry.add_agent(name=agent_name, agent_type=agent_type, command=command)
    assert agent.name == agent_name
    assert agent.agent_type == agent_type
    assert agent.command == command
    assert agent_name in registry.registry["agents"]

@pytest.mark.asyncio
async def test_remove_agent(setup_registry):
    """Test removing an agent from the registry."""
    registry = setup_registry
    agent_name = "test_agent"
    agent_type = "local"
    command = "python agent.py"
    
    await registry.add_agent(name=agent_name, agent_type=agent_type, command=command)
    assert agent_name in registry.registry["agents"]
    
    result = await registry.remove_agent(agent_name)
    assert result is True
    assert agent_name not in registry.registry["agents"]

@pytest.mark.asyncio
async def test_get_tool(setup_registry):
    """Test retrieving a tool from the registry."""
    registry = setup_registry
    tool_name = "test_tool"
    tool_type = "LOCAL"
    command = "echo 'Hello, World!'"
    
    await registry.add_tool(name=tool_name, tool_type=tool_type, command=command)
    tool = await registry.get_tool(tool_name)
    assert tool["name"] == tool_name
    assert tool["tool_type"] == tool_type
    assert tool["command"] == command

@pytest.mark.asyncio
async def test_get_agent(setup_registry):
    """Test retrieving an agent from the registry."""
    registry = setup_registry
    agent_name = "test_agent"
    agent_type = "local"
    command = "python agent.py"
    
    await registry.add_agent(name=agent_name, agent_type=agent_type, command=command)
    agent = await registry.get_agent(agent_name)
    assert agent["name"] == agent_name
    assert agent["agent_type"] == agent_type
    assert agent["command"] == command

@pytest.mark.asyncio
async def test_bulk_remove_tools(setup_registry):
    """Test bulk removal of tools."""
    registry = setup_registry
    tool_names = ["tool1", "tool2", "tool3"]
    for name in tool_names:
        await registry.add_tool(name=name, tool_type="LOCAL", command="echo 'Hello'")
    
    result = await registry.bulk_remove_tools(tool_names)
    assert all(result.values())  # All tools should be removed
    assert not any(name in registry.registry["tools"] for name in tool_names)

@pytest.mark.asyncio
async def test_bulk_remove_agents(setup_registry):
    """Test bulk removal of agents."""
    registry = setup_registry
    agent_names = ["agent1", "agent2", "agent3"]
    for name in agent_names:
        await registry.add_agent(name=name, agent_type="local", command="python agent.py")
    
    result = await registry.bulk_remove_agents(agent_names)
    assert all(result.values())  # All agents should be removed
    assert not any(name in registry.registry["agents"] for name in agent_names)

def test_registry_summary(setup_registry):
    """Test getting a summary of the registry."""
    registry = setup_registry
    summary = registry.get_registry_summary()
    assert summary["total_tools"] == 0
    assert summary["total_agents"] == 0
    assert summary["tools"] == []
    assert summary["agents"] == []
