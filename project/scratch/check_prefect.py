import asyncio
from prefect.client.orchestration import get_client

async def check():
    async with get_client() as client:
        runs = await client.read_flow_runs(limit=10)
        for r in runs:
            print(f"Run: {r.name} | State: {r.state_name} | Created: {r.created}")

if __name__ == "__main__":
    asyncio.run(check())
