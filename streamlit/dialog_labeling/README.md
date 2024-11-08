# README

本例展示了 streamlit + sqlalchemy 制作标注工具的例子, 主要的技术点如下:

- 使用 `sqlalchemy`, 这样方便对接数据库后端, 本例采用 sqlite3, 但由于使用了 `sqlalchemy` 可以很方便地迁移到 mysql 等
- `streamlit` 与数据库连接的典型做法, 也同样适用于 Flask 和 FastAPI
- `streamlit` 里 `session_state` 在 rerun 时的正确处理方式


使用方式

```python
python create_db.py  # 创建本地 sqlite3 文件
streamlit run app.py # 标注保存时将保存进数据库里
```

