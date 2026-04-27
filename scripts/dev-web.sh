#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../apps/web"
if [ ! -d node_modules ]; then
  npm ci --ignore-scripts
fi
npm run dev
