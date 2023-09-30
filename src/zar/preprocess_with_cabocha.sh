#! /bin/bash

INPUT=${1}

JUMAN_DIC="/home/your_path/local/lib/mecab/dic/jumandic"
JUMAN_DEP_MODEL="/home/your_path/local/lib/cabocha/model/dep.juman.model"
JUMAN_CHUNK_MODEL="/home/your_path/local/lib/cabocha/model/chunk.juman.model"
JUMAN_NE_MODEL="/home/your_path/local/lib/cabocha/model/ne.juman.model"

CABOCHA_CMD="/home/your_path/local/bin/cabocha -f1 -n1 -P JUMAN -d ${JUMAN_DIC} -m ${JUMAN_DEP_MODEL} -M ${JUMAN_CHUNK_MODEL} -N ${JUMAN_NE_MODEL}"

${CABOCHA_CMD} < ${INPUT}
