"""
database.py
------------
Generates a synthetic but realistic SQLite sales database for the POC.

Why this file exists on its own:
    The agents should not care HOW the data was created, only that a table
    called `sales` exists with sensible columns. Keeping data generation
    separate from agent logic means you can later swap this file for a
    real data warehouse connector without touching any agent code.

Schema (table: sales):
    id            INTEGER PRIMARY KEY
    order_date    TEXT   (YYYY-MM-DD)
    region        TEXT   (North, South, East, West)
    product       TEXT   (product name)
    category      TEXT   (product category)
    units_sold    INTEGER
    unit_price    REAL
    revenue       REAL   (units_sold * unit_price)
    discount_pct  REAL   (0.0 - 0.30)

The data spans January 2025 to December 2025 and has two intentional,
deliberate "stories" baked in so the agents (and you, when chatting with
the data) have real patterns to discover rather than pure noise:

  1. West region, Electronics category, March 2025: a sharp DROP in
     units sold (down to ~25% of normal) with elevated discounting,
     simulating something like a supply disruption or competitor issue.

  2. All regions, all categories, November-December 2025: a SPIKE in
     units sold (roughly 1.5x-1.9x normal), simulating a holiday/
     year-end seasonal surge.

This gives the agent something real to "discover" when a user asks
"Why did sales drop?" or "Why did sales spike at year end?" -- which
matches the exact business question style in the problem statement.
"""

import sqlite3
import random
from datetime import date, timedelta

DB_PATH = "sales_data.db"

REGIONS = ["North", "South", "East", "West"]

PRODUCTS = {
    "Electronics": ["Laptop", "Smartphone", "Headphones", "Smartwatch"],
    "Apparel": ["T-Shirt", "Jeans", "Jacket", "Sneakers"],
    "Home": ["Blender", "Vacuum Cleaner", "Cookware Set", "Lamp"],
    "Beauty": ["Shampoo", "Perfume", "Skincare Kit", "Makeup Kit"],
}

BASE_PRICE = {
    "Laptop": 800, "Smartphone": 600, "Headphones": 120, "Smartwatch": 200,
    "T-Shirt": 20, "Jeans": 45, "Jacket": 90, "Sneakers": 70,
    "Blender": 60, "Vacuum Cleaner": 150, "Cookware Set": 110, "Lamp": 35,
    "Shampoo": 12, "Perfume": 55, "Skincare Kit": 40, "Makeup Kit": 65,
}


def _daterange(start: date, end: date):
    days = (end - start).days
    for i in range(days + 1):
        yield start + timedelta(days=i)


def build_database(db_path: str = DB_PATH, seed: int = 42) -> str:
    """
    Creates (or recreates) the sales SQLite database with synthetic data
    spanning Jan 2025 to Jun 2025.

    Returns the path to the created database file.
    """
    random.seed(seed)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS sales")
    cur.execute(
        """
        CREATE TABLE sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_date TEXT NOT NULL,
            region TEXT NOT NULL,
            product TEXT NOT NULL,
            category TEXT NOT NULL,
            units_sold INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            revenue REAL NOT NULL,
            discount_pct REAL NOT NULL
        )
        """
    )

    start = date(2025, 1, 1)
    end = date(2025, 12, 31)

    rows = []
    for day in _daterange(start, end):
        # Not every product sells every day -- random subset per day
        for category, products in PRODUCTS.items():
            for product in products:
                for region in REGIONS:
                    # Base probability a sale happens for this combo/day
                    if random.random() > 0.35:
                        continue

                    base_units = random.randint(1, 15)
                    price = BASE_PRICE[product] * random.uniform(0.95, 1.05)
                    discount = round(random.uniform(0.0, 0.15), 2)

                    # --- Intentional anomaly 1 ---
                    # West region, Electronics category, March 2025:
                    # demand collapses (simulating a supply/competitor issue)
                    if (
                        region == "West"
                        and category == "Electronics"
                        and day.month == 3
                    ):
                        base_units = max(0, int(base_units * 0.25))
                        discount = round(random.uniform(0.15, 0.30), 2)

                    # --- Intentional anomaly 2 ---
                    # All regions/categories, Nov-Dec 2025: holiday/year-end
                    # seasonal demand surge (simulating Black Friday / festive
                    # season buying behaviour), with slightly deeper discounts
                    # typical of seasonal promotions.
                    elif day.month in (11, 12):
                        surge_factor = random.uniform(1.5, 1.9)
                        base_units = int(base_units * surge_factor)
                        discount = round(random.uniform(0.05, 0.25), 2)

                    if base_units == 0:
                        continue

                    revenue = round(base_units * price * (1 - discount), 2)

                    rows.append(
                        (
                            day.isoformat(),
                            region,
                            product,
                            category,
                            base_units,
                            round(price, 2),
                            revenue,
                            discount,
                        )
                    )

    cur.executemany(
        """
        INSERT INTO sales
            (order_date, region, product, category,
             units_sold, unit_price, revenue, discount_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    conn.commit()
    conn.close()
    return db_path


def get_schema_description(db_path: str = DB_PATH) -> str:
    """
    Returns a human/LLM-readable description of the sales table schema.
    This is what gets injected into the SQL Generator Agent's prompt so the
    LLM knows what columns exist without us hardcoding the schema in agents.py.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(sales)")
    columns = cur.fetchall()
    conn.close()

    lines = ["Table: sales", "Columns:"]
    for col in columns:
        # col = (cid, name, type, notnull, dflt_value, pk)
        lines.append(f"  - {col[1]} ({col[2]})")

    lines.append("")
    lines.append("Notes:")
    lines.append("  - order_date format is 'YYYY-MM-DD' (TEXT), spans 2025-01-01 to 2025-12-31")
    lines.append("  - region is one of: North, South, East, West")
    lines.append(f"  - category is one of: {', '.join(PRODUCTS.keys())}")
    lines.append("  - revenue = units_sold * unit_price * (1 - discount_pct)")
    return "\n".join(lines)


if __name__ == "__main__":
    path = build_database()
    print(f"Database created at: {path}")
    print()
    print(get_schema_description(path))
