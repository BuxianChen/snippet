from sqlalchemy.orm import Session
from models import LabelingItem
from typing import Optional

def create_labeling_item(
    db: Session,
    serial_no: Optional[str],
    date: Optional[str],
    customer_info: Optional[str],
    dialog_info_before: Optional[str],
    context: str,
    intent: Optional[str],
    reason: Optional[str],
):
    labeling_item = LabelingItem(
        serial_no=serial_no,
        date=date,
        customer_info=customer_info,
        dialog_info_before=dialog_info_before,
        context=context,
        intent=intent,
        reason=reason
    )
    db.add(labeling_item)
    db.commit()
    db.refresh(labeling_item)
    return labeling_item
