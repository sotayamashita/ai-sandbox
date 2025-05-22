import asyncio
import base64
import logging
import os

from agent import LLM, AgentLoop, ComputerAgent, LLMProvider
from computer import Computer
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    raise ValueError("ANTHROPIC_API_KEY is not set")


async def cua_claude_loop():
    # Initialize the CUA computer instance (macOS sandbox)
    async with Computer(
        display="1024x768",
        memory="4GB",
        cpu="4",
        os="macos",
        port=4000,
        verbosity=logging.DEBUG,
        telemetry_enabled=False,
    ) as macos_computer:
        agent = ComputerAgent(
            computer=macos_computer,
            loop=AgentLoop.ANTHROPIC,
            model=LLM(provider=LLMProvider.ANTHROPIC),
        )

        # Execute tasks in an infinite loop
        while True:
            # Take screenshot to check current state
            screenshot = await macos_computer.interface.screenshot()
            screenshot_base64 = base64.b64encode(screenshot).decode("utf-8")
            logger.info(f"Screenshot captured: {len(screenshot_base64)} bytes")

            # Get task input from user
            task = input("Enter task (type 'exit' to quit): ")
            logger.info(f"User task: {task}")

            # Check exit condition
            if task.lower() in ["exit", "quit"]:
                logger.info("Program terminated by user")
                break

            # Execute task
            try:
                # Properly consume the async generator returned by agent.run()
                async for response in agent.run(task=task):
                    # Process each response if needed
                    pass
                logger.info(f"Task completed: {task}")
            except Exception as e:
                logger.error(f"Error during task execution: {e}", exc_info=True)

            print("\nTask completed. Please enter your next task.\n")


if __name__ == "__main__":
    try:
        asyncio.run(cua_claude_loop())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
