from typing import List, Optional

VALID_SIDES = ["BUY", "SELL"]
VALID_ORDER_TYPES = ["MARKET", "LIMIT", "STOP_LIMIT", "STOP_MARKET", "TRAILING_STOP_MARKET"]

def validate_order_input(
    symbol: str,
    side: str,
    order_type: str,
    quantity: float,
    price: Optional[float] = None,
    stop_price: Optional[float] = None,
    callback_rate: Optional[float] = None
) -> List[str]:
    errors = []
    
    if side.upper() not in VALID_SIDES:
        errors.append(f"Invalid side: {side}. Must be BUY or SELL.")
    
    if order_type.upper() not in VALID_ORDER_TYPES:
        errors.append(f"Invalid order type: {order_type}. Must be one of {VALID_ORDER_TYPES}.")
        
    if quantity <= 0:
        errors.append("Quantity must be greater than 0.")
        
    if order_type.upper() in ["LIMIT", "STOP_LIMIT"] and (price is None or price <= 0):
        errors.append(f"Price is required and must be > 0 for {order_type} orders.")
        
    if order_type.upper() in ["STOP_LIMIT", "STOP_MARKET"] and (stop_price is None or stop_price <= 0):
        errors.append(f"Stop Price is required and must be > 0 for {order_type} orders.")
        
    if order_type.upper() == "TRAILING_STOP_MARKET" and (callback_rate is None or not (0.1 <= callback_rate <= 5.0)):
        errors.append("Callback Rate is required and must be between 0.1 and 5.0 for TRAILING_STOP_MARKET.")
        
    return errors
