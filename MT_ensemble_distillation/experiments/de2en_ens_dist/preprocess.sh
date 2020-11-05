#! /usr/bin/env bash

# Define the following environment variables:
#
# SRC - path to raw WMT16 data
# TGT - where to output the preprocessed data

fairseq-preprocess \
    --source-lang de --target-lang en \
    --trainpref $SRC/train.tok.clean.bpe.32000 \
    --validpref $SRC/newstest2013.tok.bpe.32000 \
    --testpref $SRC/newstest2014.tok.bpe.32000 \
    --destdir $TGT \
    --nwordssrc 32768 --nwordstgt 32768 \
    --joined-dictionary \
    --workers 20

# eof
