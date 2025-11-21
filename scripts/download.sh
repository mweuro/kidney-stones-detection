#!/bin/bash
curl -L -o kidney-stone-images.zip https://www.kaggle.com/api/v1/datasets/download/safurahajiheidari/kidney-stone-images && \
mkdir ./data && \
unzip -q kidney-stone-images.zip -d ./data && \
rm kidney-stone-images.zip
