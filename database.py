import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy import Float
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, 'faces.db')
engine = create_engine(f'sqlite:///{DB_PATH}', echo=False, future=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Face(Base):
    __tablename__ = 'faces'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    profession = Column(String, nullable=True)
    image_path = Column(String, nullable=False)


class AlertAck(Base):
    __tablename__ = 'alert_acks'
    id = Column(Integer, primary_key=True)
    camera = Column(String, nullable=True)
    track_key = Column(String, nullable=True)
    name = Column(String, nullable=True)
    ts = Column(Integer, nullable=False)


def init_db():
    Base.metadata.create_all(bind=engine)


if __name__ == '__main__':
    init_db()
    print('DB initialized at', DB_PATH)
