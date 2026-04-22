# Simplified Async-FastAPI & React Industrial Binance Futures Terminal 🚀

A production-grade, multi-modal trading terminal for the Binance Futures Testnet (USDT-M). This project features a high-performance Asynchronous Python backend powered by FastAPI, and a dense, Bloomberg-inspired React 19 Web Dashboard with a brutalist aesthetic.

## Features

- **Advanced Order Types**: Place `MARKET`, `LIMIT`, `STOP_LIMIT`, `STOP_MARKET`, and `TRAILING_STOP_MARKET` orders.
- **Robust API Layer**: Secure HMAC-SHA256 signature generation, structured request building, and complete HTTP error handling.
- **Interactive CLI**: A terminal wizard built with `rich` and `questionary` for seamless, prompt-driven order placement.
- **Industrial Web Dashboard**: A React + Tailwind CSS frontend sporting a dense, monospace, brutalist UI designed for engineers.
- **Comprehensive Logging**: Detailed `INFO` and `DEBUG` file logging for audits and debugging.

## Project Structure

```text
trading_bot/
├── bot/                # Core trading logic
│   ├── client.py       # Binance HTTP Client & Authentication
│   ├── orders.py       # API routing and payload construction
│   ├── validators.py   # Strict input validation logic
│   ├── interactive.py  # Rich/Questionary CLI wizard
│   └── logging_config.py # Log file setup
├── web/                # FastAPI backend
│   └── app.py          # API endpoints for orders and balances
├── ui/                 # React Web Dashboard
│   ├── src/            # React components (Tailwind + IBM Plex Mono)
│   └── package.json    # UI dependencies
├── cli.py              # Main CLI entry point
├── config.py           # Environment variables (.env) handler
├── requirements.txt    # Python dependencies
└── bot.log             # Auto-generated log file
```

## Setup & Installation

### 1. Python Backend Setup

Requires Python 3.10+.

```bash
# Clone the repository and navigate to the folder
cd "trading_bot"

# Create a virtual environment and activate it
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the `trading_bot` directory and add your Binance Futures **Testnet** credentials:

```env
BINANCE_API_KEY=your_testnet_api_key
BINANCE_API_SECRET=your_testnet_api_secret
```

*Note: Ensure your `.env` file is added to `.gitignore` to prevent credential leaks.*

### 3. React Frontend Setup

Requires Node.js v20.19+ or v22.12+.

```bash
cd ui
npm install
```

---

## Usage Guide

### Mode 1: Interactive CLI Wizard
To launch the guided terminal wizard, simply run the CLI without arguments:

```bash
python cli.py
```
This will launch a step-by-step menu that prompts you for the trading pair, side, order type, and amounts, finishing with an ASCII summary table.

### Mode 2: Headless CLI (Scripting)
For automation, pass the order parameters directly as arguments:

```bash
python cli.py --symbol BTCUSDT --side BUY --type LIMIT --quantity 0.001 --price 75000
```

### Mode 3: Web Dashboard
To use the visual UI, you must run both the FastAPI backend and the React development server.

1. **Start the API Server**:
   ```bash
   uvicorn web.app:app --host 127.0.0.1 --port 8000
   ```
2. **Start the React UI**:
   ```bash
   cd ui
   npm run dev
   ```
Open your browser to `http://localhost:5173`.

---

## Order Types Reference

- **MARKET**: Executes immediately at current market price. Requires `--quantity`.
- **LIMIT**: Executes at a specific price. Requires `--quantity` and `--price`.
- **STOP_MARKET**: Triggers a market order when the stop price is hit. Requires `--stop-price`.
- **TRAILING_STOP_MARKET**: Trails the price by a specific percentage. Requires `--callback-rate` (0.1 to 5.0).

> **Disclaimer**: This bot is currently configured to run **strictly on the Binance Futures Testnet**. Always verify your code on the testnet before ever considering pointing it to a live production environment.
