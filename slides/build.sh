#!/usr/bin/env bash
# Build the slide deck (PDF + HTML) with Marp.
#
# Marp needs a Chromium-based browser for PDF/PNG output.  If none is on the
# system, install a headless one once:
#   npx @puppeteer/browsers install chrome-headless-shell@stable --path /tmp/browsers
#   export CHROME_PATH=/tmp/browsers/chrome-headless-shell/*/chrome-headless-shell-linux64/chrome-headless-shell
set -euo pipefail
cd "$(dirname "$0")"

DECK=solar-panel-segmentation.md

npx --yes @marp-team/marp-cli@latest "$DECK" --html --allow-local-files -o solar-panel-segmentation.html
npx --yes @marp-team/marp-cli@latest "$DECK" --pdf  --allow-local-files -o solar-panel-segmentation.pdf
echo "Done: solar-panel-segmentation.{html,pdf}"
