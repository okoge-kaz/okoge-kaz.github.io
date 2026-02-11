.PHONY: dev build

# ローカル開発（非公開記事も表示）
dev:
	hugo server -D

# プロダクションビルド（非公開記事は除外）
build:
	hugo --minify
