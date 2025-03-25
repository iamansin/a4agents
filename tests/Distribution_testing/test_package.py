import sys
import os
import asyncio
import time 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from a4agents_core.Registry.WheelInstaller import WheelInstaller
from a4agents_core.Registry.Runner import PackageExecutor, CommunicationMode
from a4agents_core.Registry.BaseRegistry import AgentHandler
from pathlib import Path
import time

async def main():
    # installer = WheelInstaller()
 
    
    # try:
    #     start_time = time.time()
    #     # print(f"Current OS: {installer.os_type}")
    #     venv_path, executable_path = await installer.install_wheel('./tests/Distribution_testing/cdss_agent-0.1.5-py3-none-any.whl')
    #     print(f"Installed in virtual environment: {venv_path}")
    #     print(f"Total time taken to setup like pipx is : {time.time() - start_time}")
    # except Exception as e:
    #     print(f"Installation failed: {e}")
    # try:
    #     # print(f"Current OS: {installer.os_type}")
    #     deleted = await installer.cleanup_old_venvs(venv_name= 'cdss_agent_venv')
    #     # print(f"Installed in virtual environment: {venv_path}")
    # except Exception as e:
    #     print(f"Deletion failed: {e}")
    
    # Create a mock handler (in real scenario, this would be from your system)
    

    print("Now Executing the Agent-------------->")
    executor = PackageExecutor(communication_mode=CommunicationMode.REAL_TIME)
    mock_handler = AgentHandler(
        name='cdss_agent',
        agent_type= "LOCAL",
        venv_path=Path('./venvs/cdss_agent_venv'),
        entry_point=Path('./venvs/cdss_agent_venv/Scripts/Cdss_agent.exe'),
        dir = "any"
    )
    start_time = time.perf_counter()
    try:
        # Execute a package method
        result = await executor.execute_package(
            handler=mock_handler,
            args = []
        )
        
        print("Execution Result:", result)
        print(f"Total time taken to execute: {time.perf_counter() - start_time}")
        # Get performance metrics
        metrics = executor.get_metrics()
        print("Execution Metrics:", metrics)
        
    except Exception as e:
        print(f"Execution failed: {e}")

if __name__ == '__main__':
    asyncio.run(main())