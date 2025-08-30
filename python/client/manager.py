import asyncio
import aiohttp
from input_client import new_game, update_state, hero_action, update_board, solution
from typing import Callable, Awaitable, Dict, Any


class RequestCommand:
  def __init__(self, url: str, payload: Dict, callback: Callable[[Dict], None]):
    self.url = url
    self.payload = payload
    self.callback = callback

class RequestManager:
  def __init__(self, host: str):
    self.host = host
    self.queue: asyncio.Queue[RequestCommand] = asyncio.Queue()
    self.task: asyncio.Task | None = None
    self._running = False

  def start(self):
    if not self._running:
      self._running = True
      self.task = asyncio.create_task(self._run_loop())

  async def stop(self):
    self._running = False
    if self.task: await self.task

  def enqueue(self, url: str, payload: Dict, callback: Callable[[Dict], None]):
    cmd = RequestCommand(url, payload, callback)
    self.queue.put_nowait(cmd)

  async def _run_loop(self):
    async with aiohttp.ClientSession() as session:
      while self._running:
        cmd: RequestCommand = await self.queue.get()
        try:
          async with session.post(cmd.url, json=cmd.payload) as resp:
            cmd.callback(await resp.json())
        except Exception as e:
          print(f"[RequestManager] Error contacting {self.host}: {e}")
        finally:
          self.queue.task_done()
