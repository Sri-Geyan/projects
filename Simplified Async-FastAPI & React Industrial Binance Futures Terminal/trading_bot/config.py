import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_KEY = os.getenv("BINANCE_API_KEY")
    API_SECRET = os.getenv("BINANCE_API_SECRET")
    IS_TESTNET = os.getenv("IS_TESTNET", "True").lower() == "true"

    @classmethod
    def validate(cls):
        if not cls.API_KEY or not cls.API_SECRET:
            raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env file.")
