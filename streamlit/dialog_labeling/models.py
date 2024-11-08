from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class LabelingItem(Base):
    __tablename__ = 'LabelingItem'
    id = Column(Integer, primary_key=True)
    serial_no = Column(String(100), nullable=True)
    date = Column(String(20), nullable=True)
    customer_info = Column(Text, nullable=True)
    dialog_info_before = Column(Text, nullable=True)
    context = Column(Text, nullable=False)
    intent = Column(String(50), nullable=True)
    reason = Column(Text, nullable=True)