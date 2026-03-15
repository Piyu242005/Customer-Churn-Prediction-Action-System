"""
Database Integration
SQLite ORM models and initialization for churn data

Author: Piyush Ramteke
GitHub: github.com/Piyu242005/neural-network-churn
"""

import os

import pandas as pd
from sqlalchemy import Column, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


DB_FILE = os.path.join("data", "churn_data.db")
DB_URL = f"sqlite:///{DB_FILE}"

engine = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Customer(Base):
    """Raw customer transaction rows imported from CSV."""

    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    Order_ID = Column(String, index=True)
    Customer_ID = Column(String, index=True)
    Order_Date = Column(String)
    Quantity = Column(Integer)
    Unit_Price = Column(Float)
    Discount_Rate = Column(Float)
    Revenue = Column(Float)
    Cost = Column(Float)
    Profit = Column(Float)
    Region = Column(String)
    Product_Category = Column(String)
    Customer_Segment = Column(String)
    Payment_Method = Column(String)


def _resolve_csv_path(csv_path: str) -> str:
    """Resolve input CSV path for calls from workspace root or src/ directory."""
    if os.path.exists(csv_path):
        return csv_path

    src_relative = os.path.join("..", csv_path)
    if os.path.exists(src_relative):
        return src_relative

    return csv_path


def init_db(csv_path: str = "data/Business_Analytics_Dataset_10000_Rows.csv") -> int:
    """Create tables and populate customer rows from CSV if needed.

    Returns:
        int: Number of rows present in the customers table after initialization.
    """
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        existing_count = db.query(Customer).count()
        if existing_count > 0:
            print(f"Database already populated with {existing_count} rows.")
            return existing_count

        resolved_csv = _resolve_csv_path(csv_path)
        if not os.path.exists(resolved_csv):
            raise FileNotFoundError(
                f"Could not find source CSV: {csv_path}. Tried: {resolved_csv}"
            )

        print(f"Populating SQLite database from {resolved_csv}...")
        df = pd.read_csv(resolved_csv)

        if "Order_Date" in df.columns:
            df["Order_Date"] = df["Order_Date"].astype(str)

        required_columns = {
            "Order_ID",
            "Customer_ID",
            "Order_Date",
            "Quantity",
            "Unit_Price",
            "Discount_Rate",
            "Revenue",
            "Cost",
            "Profit",
            "Region",
            "Product_Category",
            "Customer_Segment",
            "Payment_Method",
        }
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV missing required columns: {missing}")

        chunk_size = 1000
        for start_idx in range(0, len(df), chunk_size):
            chunk = df.iloc[start_idx : start_idx + chunk_size]
            records = chunk.to_dict(orient="records")
            for record in records:
                clean = {k: (None if pd.isna(v) else v) for k, v in record.items()}
                db.add(Customer(**clean))
            db.commit()

        total_rows = db.query(Customer).count()
        print(f"Successfully populated database with {total_rows} rows.")
        return total_rows

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
