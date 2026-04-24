from typing import Dict, Any

def build_order_payload(
    symbol: str, 
    side: str, 
    order_type: str, 
    quantity: float, 
    price: float = None, 
    stop_price: float = None,
    callback_rate: float = None
) -> Dict[str, Any]:
    payload = {
        "symbol": symbol.upper(),
        "side": side.upper(),
        "type": order_type.upper(),
        "quantity": quantity
    }
    
    if order_type.upper() == "LIMIT":
        payload["price"] = price
        payload["timeInForce"] = "GTC"
    elif order_type.upper() == "STOP_LIMIT":
        payload["price"] = price
        payload["stopPrice"] = stop_price
        payload["timeInForce"] = "GTC"
    elif order_type.upper() == "STOP_MARKET":
        payload["stopPrice"] = stop_price
    elif order_type.upper() == "TRAILING_STOP_MARKET":
        payload["callbackRate"] = callback_rate
        
    return payload
