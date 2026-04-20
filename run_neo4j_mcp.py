 
import asyncio
from openai import AsyncAzureOpenAI

from agents.mcp.server import MCPServerStdio
from agents import set_default_openai_client

from agents import Agent, Runner
import ssl,httpx
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

AZURE_OPENAI_API_KEY="f0e56505e5594c9db822a6e7addc7209"
AZURE_OPENAI_ENDPOINT="https://ims-openai-qa.openai.azure.com/"
AZURE_OPENAI_API_VERSION="2024-02-15-preview"
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


azure_client = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    http_client=httpx.AsyncClient(verify=False)
)
set_default_openai_client(azure_client)
print("✅ Azure OpenAI client configured for Agents SDK")

async def main():
    async with MCPServerStdio(
        params={
            "command": "mcp-neo4j-memory",
            "args": [],
            "env": {
                "NEO4J_URI": "neo4j://awsaidnval000z.jnj.com:7687",
                "NEO4J_USERNAME": "neo4j",
                "NEO4J_PASSWORD": "MVP2026!",
                "NEO4J_DATABASE": "newone",
            },
        },
        client_session_timeout_seconds=30,
    ) as mcp_server:
        
        tools = await mcp_server.list_tools()
        
        print("Tools count:", len(tools))
    async def test_model():
        try:
            response = await azure_client.chat.completions.create(
            model="iMS-GPT4oQA",
            messages=[{"role": "user", "content": "test"}]
        )
            print("✅ Model accessible:", response.choices.message.content[:50])
        except Exception as e:
            print("❌ Model error:", e)
        
        print("First 10 tool names:", [t.name for t in tools[:10]])
        model_find=await test_model()
        print("wether model is working:", model_find)
    async def mcp_agent_query(query: str, prior_context: str = ""):
        async with mcp_server:
            agent = Agent(
                        name="GraphRAG-MCP-Agent",
            instructions=f"""Biomedical expert with GraphRAG + Neo4j Memory.
            
Use MCP tools:
- create_entities/add_observations: Store query facts/entities
- search_memories/find_memories_by_name: Recall prior context
- read_graph: Graph queries (ADA trough, cohorts)
            
Prior context: {prior_context}
            
Query: {query}
            
Combine with your GraphRAG knowledge for comprehensive answers.""",
            model="iMS_GPT4o_QA",  
            mcp_servers=[mcp_server],  # Enables all 9 tools
        )
            result = await Runner.run(agent, query)
            return result.final_output

# Test multi-turn memory
    print(await mcp_agent_query("Subjects who first became ADA at week 16: store key stats (mean trough week 36)."))
    print(await mcp_agent_query("Recall ADA week 16 subjects' trough stats from memory."))




    print("MCP server started and connected")

if __name__ == "__main__":
    asyncio.run(main())
