#!/bin/bash
# Download English corpus for CBOW training

DATA_DIR="data"

mkdir -p "$DATA_DIR"

echo "=== Option 1: Penn Treebank (Recommended, ~1M tokens) ==="
if [ ! -f "$DATA_DIR/ptb.train.txt" ]; then
    echo "Downloading PTB train..."
    wget -q https://download.pytorch.org/tutorial/data/ptb.train.txt -O "$DATA_DIR/ptb.train.txt"
    wget -q https://download.pytorch.org/tutorial/data/ptb.valid.txt -O "$DATA_DIR/ptb.valid.txt"
    wget -q https://download.pytorch.org/tutorial/data/ptb.test.txt -O "$DATA_DIR/ptb.test.txt"
    echo "PTB downloaded!"
else
    echo "PTB already exists."
fi

echo ""
echo "=== Option 2: WikiText-2 (Larger, ~2M tokens) ==="
if [ ! -f "$DATA_DIR/wikitext-2-raw.txt" ]; then
    echo "Downloading WikiText-2..."
    wget -q https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip -O "$DATA_DIR/wikitext.zip"
    unzip -q "$DATA_DIR/wikitext.zip" -d "$DATA_DIR/"
    rm "$DATA_DIR/wikitext.zip"
    echo "WikiText-2 downloaded!"
else
    echo "WikiText-2 already exists."
fi

echo ""
echo "=== Download complete! ==="
echo "Files in $DATA_DIR:"
ls -lh "$DATA_DIR"
