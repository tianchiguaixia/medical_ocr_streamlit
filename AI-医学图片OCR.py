# -*- coding: utf-8 -*-
# time: 2022/10/17 11:22
# file: AI-医学图片OCR.py



import streamlit as st

from ocr.ocr import detect, recognize
from ocr.utils import bytes_to_numpy
import pandas as pd

import os
import cv2
from paddleocr import PPStructure,draw_structure_result,save_structure_res

st.title("AI-医学图片OCR")
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("gbk")


# 上传图片
uploaded_file = st.sidebar.file_uploader('请选择一张图片', type=['png', 'jpg', 'jpeg'])
print('uploaded_file:', uploaded_file)
table_engine = PPStructure(show_log=True)
if uploaded_file is not None:
    # To read file as bytes:
    # content = cv2.imread(uploaded_file)
    # st.write(content)
    bytes_data = uploaded_file.getvalue()
    # 转换格式
    img = bytes_to_numpy(bytes_data, channels='RGB')
    option_task = st.sidebar.radio('请选择要执行的任务', ('查看原图', '文本检测'))
    if option_task == '查看原图':
        st.image(img, caption='原图')
    elif option_task == '文本检测':
        im_show = detect(img)
        st.image(im_show, caption='文本检测后的图片')

    base_path="streamlit_data"

    path=os.path.exists(base_path+"/"+uploaded_file.name.split('.')[0])

    if st.button('✨ 启动!'):
        local_path=base_path +"/"+uploaded_file.name.split('.')[0]
        result = table_engine(img)
        save_structure_res(result, base_path,uploaded_file.name.split('.')[0])
        with st.container():
            with st.expander(label="json结果展示", expanded=False):
                st.write(result)
            for i in os.listdir(local_path):
                if ".xlsx" in i:
                    df = pd.read_excel(os.path.join(local_path, i))
                    df=df.fillna("")
                    st.write(df)
                    csv = convert_df(df)
                    st.download_button(
                        label="Download data as csv",
                        data=csv,
                        file_name='large_df.csv',
                        mime='text/csv',
                    )





