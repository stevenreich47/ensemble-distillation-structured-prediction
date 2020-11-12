#! /usr/bin/env bash

# Define the following environment variables:
#
# RAW_DATA - path to raw WMT16 data
# DATA_DIR - where to output the preprocessed data

fairseq-preprocess \
    --source-lang de --target-lang en \
    --trainpref $RAW_DATA/train.tok.clean.bpe.32000 \
    --validpref $RAW_DATA/newstest2013.tok.bpe.32000 \
    --testpref $RAW_DATA/newstest2014.tok.bpe.32000 \
    --destdir $DATA_DIR \
    --nwordssrc 32768 --nwordstgt 32768 \
    --joined-dictionary \
    --workers 20

# eof
