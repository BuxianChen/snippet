import streamlit as st
import uuid
from datetime import datetime
import re

from connect_db import get_db
from crud import create_labeling_item

if "serial_no" not in st.session_state:
    st.session_state["serial_no"] = ""

if "serial_date" not in st.session_state:
    st.session_state["serial_date"] = ""

if "dialog" not in st.session_state:
    st.session_state["dialog"] = []


def rebuild_dialog():
    new_dialog = []
    if st.session_state["dialog"] and st.session_state["dialog"][-1]["content"] == "":
        last_empty = [st.session_state["dialog"][-1]]
    else:
        last_empty = []

    for item in st.session_state["dialog"]:
        role = item["role"]
        text = item["content"]
        for line in re.split(r"\r?\n", text):
            line = line.strip()
            if not line:
                continue
            match_result = re.match("(客户|用户|坐席|客服)\s*[:：]?\s*(.*)", line)
            if match_result:
                # print("match", line)
                role = match_result.group(1)
                line = match_result.group(2)
                role = "客户" if role in ["客户", "用户"] else "坐席"
                new_dialog.append({"role": role, "content": line})
            else:
                assert role in ["客户", "坐席"]
                new_dialog.append({"role": role, "content": line})
    
    st.session_state["dialog"] = new_dialog
    st.session_state["dialog"].extend(last_empty)

    for idx, item in enumerate(st.session_state["dialog"]):
        st.session_state[f"content_{idx}"] = item["content"]
        st.session_state[f"role_{idx}"] = item["role"]


def delete_dialog(idx):
    st.session_state["dialog"].pop(idx)
    rebuild_dialog()

def add_dialog():
    st.session_state["dialog"].append({"role": "客户", "content": ""})
    rebuild_dialog()

def modify_dialog_content(idx):
    st.session_state["dialog"][idx]["content"] = st.session_state[f"content_{idx}"]
    rebuild_dialog()

def modify_dialog_role(idx):
    st.session_state["dialog"][idx]["role"] = st.session_state[f"role_{idx}"]
    rebuild_dialog()

def save_to_db():
    rebuild_dialog()

    context = "\n".join([
        item["role"]+":"+item["content"]
        for item in st.session_state["dialog"]
    ])

    with get_db() as db:
        # db = next(get_db())
        create_labeling_item(
            db=db,
            serial_no=st.session_state["serial_no"],
            date=st.session_state["date"],
            customer_info=st.session_state["customer_info"],
            dialog_info_before=st.session_state["dialog_info_before"],
            context=context,
            intent=st.session_state["intent"],
            reason=st.session_state["reason"]
        )

    for key in ["serial_no", "date", "customer_info", "dialog_info_before", "intent", "reason"]:
        st.session_state[key] = ""
    st.session_state["dialog"] = []


st.text_input("录音流水号", key="serial_no")
st.text_input("通话日期", key="date")
st.text_input("客户额外信息", key="customer_info")
st.text_input("前序对话信息", key="dialog_info_before")


rebuild_dialog()
for idx, (role, text) in enumerate(st.session_state["dialog"]):
    columns = st.columns([1, 5, 1])
    with columns[0]:
        st.selectbox("角色", ["客户", "坐席"], key=f"role_{idx}", on_change=modify_dialog_role, args=(idx,))
    with columns[1]:
        st.text_area("文本", key=f"content_{idx}", on_change=modify_dialog_content, args=(idx,))
    with columns[2]:
        st.button("删除", key=f"delete_{idx}", on_click=delete_dialog, args=(idx,))

st.button("增加内容", on_click=add_dialog)

st.text_input("意图", key="intent")
st.text_input("标注理由", key="reason")
st.button("保存至数据库", on_click=save_to_db)
