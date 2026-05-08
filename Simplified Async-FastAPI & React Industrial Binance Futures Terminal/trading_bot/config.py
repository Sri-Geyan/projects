import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_KEY = os.getenv("BINANCE_API_KEY") or input("Enter BINANCE_API_KEY: ")
    API_SECRET = os.getenv("BINANCE_API_SECRET") or input("Enter BINANCE_API_SECRET: ")
    _testnet_env = os.getenv("IS_TESTNET")
    _testnet_input = _testnet_env if _testnet_env is not None else input("Use Testnet? (True/False) [True]: ") or "True"
    IS_TESTNET = _testnet_input.lower() == "true"

    @classmethod
    def validate(cls):
        if not cls.API_KEY or not cls.API_SECRET:
            raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET must be provided.")
