import os
import tempfile

# Never let a developer's real .env or real saved history leak into the test run.
os.environ["WORDLE_NO_DOTENV"] = "1"
os.environ["WORDLE_MEMORY_PATH"] = os.path.join(tempfile.mkdtemp(prefix="wordle_test_mem_"), "history.json")
