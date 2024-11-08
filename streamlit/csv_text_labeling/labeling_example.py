import os

import pandas as pd
import streamlit as st


def read_data(data_path):
    if not os.path.exists(data_path):
        with open(data_path, "w") as fw:
            fw.write(",".join(["id", "text", "label"])+"\n")
    df = pd.read_csv(data_path, dtype=str).fillna("")
    if "label" not in df.columns:
        df["label"] = ""
    return df

def get_unlabeled_indexes(input_df, output_df):
    labeled_ids = set(output_df["id"].to_numpy().tolist())
    unlabeled_indexes = []
    for i, uniq_id in enumerate(input_df["id"]):
        if uniq_id not in labeled_ids:
            unlabeled_indexes.append(i)
    return unlabeled_indexes


st.set_page_config(
    page_title="对话片段标注工具",
    page_icon='📌',
    layout="wide"
)


input_path = "text.csv"
output_path = "result.csv"
LABELS = ["好", "中", "差"]

if "input_df" not in st.session_state:
    st.session_state["input_df"] = read_data(input_path)
if "output_df" not in st.session_state:
    st.session_state["output_df"] = read_data(output_path)
if "unlabeled_indexes" not in st.session_state:
    st.session_state["unlabeled_indexes"] = get_unlabeled_indexes(
        st.session_state["input_df"],
        st.session_state["output_df"]
    )
if "current_index" not in st.session_state:
    st.session_state["current_index"] = st.session_state["unlabeled_indexes"][0] if st.session_state["unlabeled_indexes"] else -1


if st.session_state["current_index"] == -1:
    st.write()


######################### 页面定义区（侧边栏） ########################
st.sidebar.title('📌 标注平台')
st.sidebar.write(f'input_path: {input_path}')
st.sidebar.write(f'output_path: {output_path}')

label_tab, dataset_tab = st.tabs(['Label', 'Dataset'])
######################### 页面定义区（标注页面） ########################
with label_tab:
    if st.session_state["current_index"] == -1:
        st.write("标注已完成, 请前往数据集页面查看已标注数据并保存")

    else:
        st.write("待标注句子：")
        column = st.columns([1])[0]
        with column:
            st.write(st.session_state["input_df"].iloc[st.session_state["current_index"]]["text"])
        label = st.selectbox(f'标注结果', LABELS)

        save_button = st.button('保存')
        if save_button:
            st.session_state["unlabeled_indexes"].pop(0)
            with open(output_path, "a") as fw:
                row = st.session_state["input_df"].iloc[st.session_state["current_index"]]
                fw.write(",".join([row["id"], row["text"], label])+"\n")
            st.session_state["current_index"] = st.session_state["unlabeled_indexes"][0] if st.session_state["unlabeled_indexes"] else -1
            st.experimental_rerun()

######################### 页面定义区（数据集页面） #######################
with dataset_tab:
    rank_texts_list = []
    df = read_data(output_path)
    st.dataframe(df)
