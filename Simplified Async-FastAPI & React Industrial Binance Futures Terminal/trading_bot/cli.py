import asyncio
import questionary
from rich.console import Console
from rich.table import Table
from bot.client import BinanceClient
from bot.orders import build_order_payload
from bot.validators import validate_order_input
from config import Config

console = Console()

async def interactive_wizard():
    console.print("[bold cyan]🚀 Binance Futures Terminal - Interactive Wizard[/bold cyan]\n")
    
    try:
        Config.validate()
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return

    client = BinanceClient(Config.API_KEY, Config.API_SECRET, Config.IS_TESTNET)
    
    symbol = await questionary.text("Trading Pair (e.g., BTCUSDT):", default="BTCUSDT").ask_async()
    side = await questionary.select("Side:", choices=["BUY", "SELL"]).ask_async()
    order_type = await questionary.select("Order Type:", choices=["MARKET", "LIMIT", "STOP_LIMIT", "STOP_MARKET", "TRAILING_STOP_MARKET"]).ask_async()
    quantity = float(await questionary.text("Quantity:").ask_async())
    
    price = None
    stop_price = None
    callback_rate = None
    
    if order_type in ["LIMIT", "STOP_LIMIT"]:
        price = float(await questionary.text("Limit Price:").ask_async())
    if order_type in ["STOP_LIMIT", "STOP_MARKET"]:
        stop_price = float(await questionary.text("Stop Price:").ask_async())
    if order_type == "TRAILING_STOP_MARKET":
        callback_rate = float(await questionary.text("Callback Rate (0.1 - 5.0):").ask_async())

    errors = validate_order_input(symbol, side, order_type, quantity, price, stop_price, callback_rate)
    if errors:
        for err in errors:
            console.print(f"[bold red]Validation Error:[/bold red] {err}")
        return

    payload = build_order_payload(symbol, side, order_type, quantity, price, stop_price, callback_rate)
    
    # Summary Table
    table = Table(title="Order Summary")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="magenta")
    for k, v in payload.items():
        table.add_row(k, str(v))
    console.print(table)
    
    confirm = await questionary.confirm("Execute Order?").ask_async()
    if confirm:
        with console.status("[bold green]Executing Order..."):
            result = await client.place_order(payload)
            console.print_json(data=result)

if __name__ == "__main__":
    asyncio.run(interactive_wizard())
