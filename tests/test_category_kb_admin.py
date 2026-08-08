from app.recall.category_kb_admin import (
    build_index_body,
    build_pipeline_body,
)


def test_category_index_mapping_matches_embedding() -> None:
    body = build_index_body(384)
    settings = body["settings"]["index"]
    properties = body["mappings"]["properties"]
    vector = properties["content_vector"]

    assert settings["knn"] is True
    assert vector["type"] == "knn_vector"
    assert vector["dimension"] == 384
    assert vector["method"]["space_type"] == "innerproduct"
    assert properties["summary"]["analyzer"] == "standard"


def test_category_pipeline_weights_match_two_queries() -> None:
    body = build_pipeline_body()
    processor = body["phase_results_processors"][0][
        "normalization-processor"
    ]
    weights = processor["combination"]["parameters"][
        "weights"
    ]

    assert weights == [0.7, 0.3]
    assert sum(weights) == 1.0

