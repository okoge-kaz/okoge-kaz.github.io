#!/usr/bin/env bash
#
# Toggle a blog post between draft and published.
#
# Usage:
#   ./scripts/toggle-draft.sh <post-name>
#
# Examples:
#   ./scripts/toggle-draft.sh gpt-oss-nemo-training     # toggle draft status
#   ./scripts/toggle-draft.sh gpt-oss-nemo-training publish  # force publish
#   ./scripts/toggle-draft.sh gpt-oss-nemo-training draft    # force draft
#
# To list all drafts:
#   ./scripts/toggle-draft.sh --list

set -euo pipefail

BLOG_DIR="$(cd "$(dirname "$0")/.." && pwd)/content/blog"

# --list: show all posts and their draft status
if [[ "${1:-}" == "--list" ]]; then
  echo "Blog posts:"
  for dir in "$BLOG_DIR"/*/; do
    [ -d "$dir" ] || continue
    name="$(basename "$dir")"
    file="$dir/index.md"
    [ -f "$file" ] || continue
    if grep -q '^draft: true' "$file"; then
      echo "  [DRAFT]     $name"
    else
      echo "  [PUBLISHED] $name"
    fi
  done
  exit 0
fi

if [[ -z "${1:-}" ]]; then
  echo "Usage: $0 <post-name> [publish|draft]"
  echo "       $0 --list"
  exit 1
fi

POST_NAME="$1"
ACTION="${2:-toggle}"
POST_FILE="$BLOG_DIR/$POST_NAME/index.md"

if [[ ! -f "$POST_FILE" ]]; then
  echo "Error: Post not found: $POST_FILE"
  echo ""
  echo "Available posts:"
  for dir in "$BLOG_DIR"/*/; do
    [ -d "$dir" ] || continue
    echo "  $(basename "$dir")"
  done
  exit 1
fi

HAS_DRAFT=$(grep -c '^draft: true' "$POST_FILE" || true)

case "$ACTION" in
  publish)
    if [[ "$HAS_DRAFT" -gt 0 ]]; then
      sed -i '' '/^draft: true$/d' "$POST_FILE"
      echo "Published: $POST_NAME"
    else
      echo "Already published: $POST_NAME"
    fi
    ;;
  draft)
    if [[ "$HAS_DRAFT" -eq 0 ]]; then
      sed -i '' '/^---$/,/^---$/{
        /^title:/a\
draft: true
      }' "$POST_FILE"
      echo "Set to draft: $POST_NAME"
    else
      echo "Already a draft: $POST_NAME"
    fi
    ;;
  toggle)
    if [[ "$HAS_DRAFT" -gt 0 ]]; then
      sed -i '' '/^draft: true$/d' "$POST_FILE"
      echo "Published: $POST_NAME"
    else
      sed -i '' '/^---$/,/^---$/{
        /^title:/a\
draft: true
      }' "$POST_FILE"
      echo "Set to draft: $POST_NAME"
    fi
    ;;
  *)
    echo "Unknown action: $ACTION (use publish, draft, or omit to toggle)"
    exit 1
    ;;
esac
