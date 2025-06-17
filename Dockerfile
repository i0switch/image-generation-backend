# ベースイメージとしてPython 3.10を指定
FROM python:3.10-slim

# 作業ディレクトリを設定
WORKDIR /code

# 必要なライブラリをインストールするための準備
# OpenCVなど、aptで必要なパッケージがあればここでインストール
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# requirements.txtをコンテナにコピー
COPY ./requirements.txt /code/requirements.txt

# pipでPythonライブラリをインストール
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# アプリケーションコードをコンテナにコピー
COPY ./app /code/app

# アプリケーションの起動コマンド
# Uvicornサーバーを0.0.0.0:7860で起動
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]