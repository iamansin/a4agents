from langgraph.graph import StateGraph, END

class CDSS_Agnet:
    
    def __init__(self, llm_dict :dict, tools_dict :dict):
        self._llm_dict = llm_dict
        self._tool_dict = tools_dict
        self.graph = self.compile_graph()
        
    def complie_graph(self):
        graph_builder = StateGraph()
        graph_builder.add_node()