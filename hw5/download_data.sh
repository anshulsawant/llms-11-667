#!/bin/bash

ROOT=https://data.commoncrawl.org/crawl-data/CC-MAIN-2018-17/segments/1524125937193.1/

curl -o data.warc.gz ${ROOT}warc/CC-MAIN-20180420081400-20180420101400-00000.warc.gz
gunzip data.warc.gz

curl -o data.wet.gz ${ROOT}wet/CC-MAIN-20180420081400-20180420101400-00000.warc.wet.gz
gunzip data.wet.gz

curl -o data.wat.gz ${ROOT}wat/CC-MAIN-20180420081400-20180420101400-00000.warc.wat.gz
gunzip data.wat.gz