"""
Save translation history to SQLite.
Each record includes: original text, translation, source (selection/ocr), timestamp.
"""
import sqlite3
from pathlib import Path
from datetime import datetime


class TranslationHistory:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "history.db")
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_text TEXT NOT NULL,
                    translated TEXT NOT NULL,
                    method     TEXT DEFAULT 'selection',
                    engine     TEXT DEFAULT 'en-vi',
                    created_at TEXT NOT NULL
                )
            """)
            # Index for fast searching
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created
                ON history(created_at DESC)
            """)

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def save(self, source_text: str, translated: str, method: str = "selection", engine: str = "en-vi"):
        """Save a translation. Call this after every successful translation."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO history (source_text, translated, method, engine, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (source_text.strip(), translated.strip(), method, engine, now))

    def search(self, keyword: str, limit: int = 50) -> list[dict]:
        """Search by keyword in the original text or translation."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM history "
                "WHERE source_text LIKE ? OR translated LIKE ? "
                "ORDER BY created_at DESC LIMIT ?",
                (f"%{keyword}%", f"%{keyword}%", limit)
            ).fetchall()
        return [dict(r) for r in rows]

    def recent(self, limit: int = 50) -> list[dict]:
        """Get the most recent translations."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM history ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, record_id: int):
        with self._conn() as conn:
            conn.execute("DELETE FROM history WHERE id = ?", (record_id,))

    def clear_all(self):
        with self._conn() as conn:
            conn.execute("DELETE FROM history")

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
