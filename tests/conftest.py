import os

# Never let a developer's real .env leak a key into the test run (tests that need a key set it explicitly).
os.environ["WORDLE_NO_DOTENV"] = "1"
