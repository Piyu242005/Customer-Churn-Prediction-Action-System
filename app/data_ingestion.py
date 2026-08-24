from __future__ import annotations

import io
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ["Quantity", "Unit_Price", "Discount_Rate", "Revenue", "Cost", "Profit"]


def detect_type(name: str) -> str:
    suffix = Path(name).suffix.lower()
    return {".csv": "CSV", ".xlsx": "Excel", ".xls": "Excel", ".pdf": "PDF"}.get(suffix, "Unsupported")


def read_customer_file(uploaded_file, sheet_name=0):
    kind = detect_type(uploaded_file.name)
    raw = uploaded_file.getvalue()
    if kind == "CSV":
        return pd.read_csv(io.BytesIO(raw)), kind
    if kind == "Excel":
        book = pd.ExcelFile(io.BytesIO(raw))
        return pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name), kind
    if kind == "PDF":
        try:
            import pdfplumber
        except ImportError as exc:
            raise RuntimeError("PDF support requires pdfplumber. Add it to requirements.txt.") from exc
        tables = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables() or []:
                    if table and len(table) > 1:
                        tables.append(pd.DataFrame(table[1:], columns=table[0]))
        if not tables:
            raise ValueError("No structured customer table was found in this PDF. Upload a table-based PDF, CSV, or Excel file.")
        return pd.concat(tables, ignore_index=True), kind
    raise ValueError("Unsupported file. Upload CSV, XLSX, XLS, or PDF.")


def suggest_mapping(columns):
    aliases = {
        "Quantity": ["quantity", "qty", "units", "order_quantity"],
        "Unit_Price": ["unit_price", "unit price", "price", "unitprice"],
        "Discount_Rate": ["discount_rate", "discount rate", "discount", "discount_pct"],
        "Revenue": ["revenue", "sales", "sales_amount", "total_sales"],
        "Cost": ["cost", "expense", "total_cost"],
        "Profit": ["profit", "margin", "profit_amount"],
    }
    normalized = {str(c).strip().lower().replace("-", "_"): c for c in columns}
    mapping = {}
    for target, names in aliases.items():
        for alias in names:
            key = alias.lower().replace("-", "_")
            if key in normalized:
                mapping[target] = normalized[key]
                break
    return mapping


def validate_and_normalize(df, mapping=None):
    mapping = mapping or suggest_mapping(df.columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in mapping]
    if missing:
        return None, {"missing": missing, "found": list(df.columns), "mapping": mapping}
    out = df.rename(columns={source: target for target, source in mapping.items()}).copy()
    for col in REQUIRED_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    invalid = {c: int(out[c].isna().sum()) for c in REQUIRED_COLUMNS if out[c].isna().any()}
    return out, {"missing": [], "invalid": invalid, "mapping": mapping}
