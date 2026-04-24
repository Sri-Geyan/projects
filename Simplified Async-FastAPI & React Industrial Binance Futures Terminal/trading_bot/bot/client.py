import hmac, hashlib, time, aiohttp, asyncio
from typing import Dict, Any

class BinanceClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://fapi.binance.com" if not testnet else "https://testnet.binancefuture.com"

    def _generate_signature(self, query_string: str) -> str:
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    async def _request(self, method: str, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        if params is None:
            params = {}
        
        params["timestamp"] = int(time.time() * 1000)
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        signature = self._generate_signature(query_string)
        
        url = f"{self.base_url}{endpoint}?{query_string}&signature={signature}"
        headers = {"X-MBX-APIKEY": self.api_key}

        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, headers=headers) as response:
                return await response.json()

    async def get_balance(self):
        return await self._request("GET", "/fapi/v2/balance")

    async def place_order(self, order_params: Dict[str, Any]):
        return await self._request("POST", "/fapi/v1/order", order_params)
