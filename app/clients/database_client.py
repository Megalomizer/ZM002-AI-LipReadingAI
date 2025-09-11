import sqlite3
import cv2
import numpy
from typing import List

from core.constants import DATABASE_URL
from entities.db_model import DBModel


class DatabaseClient:
    def __init__(self):
        print(DATABASE_URL)
        self.conn = sqlite3.connect(str(DATABASE_URL))
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self) -> None:
        """
        Creates required database tables if they don't exist.
        Creates two tables:
        - labels: Stores label information with id and label text
        - frames: Stores frame data with object references and binary frame data

        :return: None
        """
        
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL
        );
        """)

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS frames (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label_id INTEGER NOT NULL,
            frame_data BLOB NOT NULL,
            FOREIGN KEY (label_id) REFERENCES labels (id)
        );
        """)

        self.conn.commit()

    def insert_object(self, model: DBModel) -> int:
        """
        Insert a new object into the database.

        :param model: DBModel
        :return: id
        """

        # insert label
        self.cursor.execute("INSERT INTO labels (label) VALUES (?)", (model.label,))
        label_id = self.cursor.lastrowid

        # insert frames
        for frame in model.frames:
            _, buffer = cv2.imencode(".jpg", frame)
            self.cursor.execute("INSERT INTO frames (label_id, frame_data) VALUES (?, ?)", (label_id, buffer.tobytes()))
        self.conn.commit()
        return label_id

    def get_object(self, object_id: int) -> DBModel:
        """
        Get the object with the given id.

        :param object_id: id
        :return: DBModel
        """
        self.cursor.execute("SELECT label FROM labels WHERE id = ?", (object_id,))
        label = self.cursor.fetchone()[0]

        self.cursor.execute("SELECT label_id, frame_data FROM frames WHERE label_id = ?", (object_id,))
        rows = self.cursor.fetchall()
        frames = []
        for (frame_data,) in rows:
            frame_array = numpy.frombuffer(frame_data, dtype=numpy.uint8)
            frames.append(cv2.imdecode(frame_array, cv2.IMREAD_COLOR))

        return DBModel(label, frames)

    def close(self):
        """
        Close the database connection.
        """
        self.conn.close()
