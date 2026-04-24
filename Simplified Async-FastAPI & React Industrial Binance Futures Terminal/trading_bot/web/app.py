from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from bot.client import BinanceClient
from bot.orders import build_order_payload
from bot.validators import validate_order_input
from config import Config

app = FastAPI(title="Binance Futures API")

class OrderRequest(BaseModel):
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    callback_rate: Optional[float] = None

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/balances")
async def get_balances():
    client = BinanceClient(Config.API_KEY, Config.API_SECRET, Config.IS_TESTNET)
    return await client.get_balance()

@app.post("/order")
async def place_order(order: OrderRequest):
    errors = validate_order_input(
        order.symbol, order.side, order.order_type, 
        order.quantity, order.price, order.stop_price, order.callback_rate
    )
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})
    
    payload = build_order_payload(
        order.symbol, order.side, order.order_type, 
        order.quantity, order.price, order.stop_price, order.callback_rate
    )
    
    client = BinanceClient(Config.API_KEY, Config.API_SECRET, Config.IS_TESTNET)
    result = await client.place_order(payload)
    return result
