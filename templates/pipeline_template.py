from typing import Optional
class Pipeline:
    async def __call__(self, task_description: str, metadata: Optional[dict] = None) -> str:
        pass
