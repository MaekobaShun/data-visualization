@echo off
cd /d C:\Python_project\Data_visualization\data_vis.py
call .venv\Scripts\activate
streamlit run data_vis.py
pause