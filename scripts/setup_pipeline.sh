#!/usr/bin/env sh
set -eu
curl -X PUT "${OPENSEARCH_URL:-http://localhost:9200}/_search/pipeline/globex_hybrid_pipeline" \
  -H 'Content-Type: application/json' \
  -d '{
    "description": "KNN + BM25 双路召回归一与加权融合",
    "phase_results_processors": [{
      "normalization-processor": {
        "normalization": {"technique": "min_max"},
        "combination": {
          "technique": "arithmetic_mean",
          "parameters": {"weights": [0.7, 0.3]}
        }
      }
    }]
  }'
