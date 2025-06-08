# Nekognition（猫検出アプリケーション）

[Amazon Rekognition](https://aws.amazon.com/jp/rekognition/)のPoCのために開発しました。  
将来的に、猫の画像投稿アプリの開発を始めるとして（まじでやるかは置いといて）、

  - 画像に猫が映っていなければ、投稿させない
  - 人の顔が映っていた場合、該当箇所にモザイク処理をする  

といった機能の実装にRekognitionが使えそうか試してみました。

## 主な機能
Streamlitを使った簡易的なWebアプリケーション
- 画像のアップロード
- 画像内の猫を自動検出し、枠線で強調表示 ＊枠線のハイライトON/OFFを切り替え可能
- 顔検出領域に自動でモザイク処理
- 開発用のダミーデータによる自動テスト

## 動作環境
- [devcontainer](https://containers.dev/)（VS Code推奨）
- 必要なパッケージは`pyproject.toml`/`uv.lock`で管理

## 起動方法
1. **AWS認証情報の準備**
   - ホストOSの`~/.aws/credentials`に有効なAWSキーを設定
   - 必要なIAM権限は以下の通り

   ```json
    {
        "Version": "2012-10-17",
        "Statement": [
            {
            "Effect": "Allow",
            "Action": [
                "rekognition:DetectLabels",
                "rekognition:DetectFaces"
            ],
            "Resource": "*"
            }
        ]
    }
   ```

1. **devcontainerで開発環境を起動**
   - VS Codeで「Remote-Containers: Open Folder in Container」を実行
1. **アプリの起動**
   ```sh
   uv run streamlit run app/main.py
   ```
   - ブラウザで `http://localhost:8501` にアクセス

## テスト実行方法
1. **pytestによる自動テスト**

   ```sh
   uv run pytest
   ```

## ディレクトリ構成
- `app/` ... Streamlitアプリ本体
- `lib/` ... 画像処理・APIラッパ・ユーティリティ
- `tests/` ... 各種テストコード・テスト用画像

## 注意事項
- Amazon Rekognitionの利用にはAPI利用料が発生します。
    - https://aws.amazon.com/jp/rekognition/pricing/
- プロキシ環境下ではホストOS側でHTTP_PROXY等の環境変数を設定してください。devcontainerへ引き継ぐ設定になっています。

## 参考
- [クラス図](doc/class_diagram.pu)