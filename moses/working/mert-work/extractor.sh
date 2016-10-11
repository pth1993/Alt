#!/usr/bin/env bash
cd /home/hoang/Desktop/Alt+/moses/working/mert-work
/home/hoang/mosesdecoder/bin/extractor --sctype BLEU --scconfig case:true  --scfile run9.scores.dat --ffile run9.features.dat -r /home/hoang/Desktop/Alt+/moses/data/tune/clean.tune.tag -n run9.best100.out.gz
