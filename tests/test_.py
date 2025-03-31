import ray
import multiprocessing
import time
@ray.remote
class BaseRegistry:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name, tool):
        self.tools[name] = tool

    def get_tool(self, name):
        return self.tools.get(name, None)

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Create a global registry
registry = BaseRegistry.remote()

# Example usage in agents
def agent_process():
    strt = time.pref_counter()
    ray.get(registry.register_tool.remote("search", lambda x: f"Searching {x}"))
    tool = ray.get(registry.get_tool.remote("search"))
    print("Agent using tool:", tool("Python"))

if __name__ == "__main__":
    p1 = multiprocessing.Process(target=agent_process)
    p2 = multiprocessing.Process(target=agent_process)
 
    p1.start()
    p2.start()
    p1.join()
    p2.join()
