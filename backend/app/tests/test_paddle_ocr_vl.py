from __future__ import annotations

from app.vision.paddle_ocr_vl import _blocks_from_payload


def test_blocks_from_paddle_ocr_vl_payload() -> None:
    payload = {
        "parsing_res_list": [
            {
                "block_label": "text",
                "block_content": "学校 がっこう 학교",
                "block_bbox": [10, 20, 200, 60],
                "block_order": 1,
            }
        ]
    }

    blocks = _blocks_from_payload(payload)

    assert len(blocks) == 1
    assert blocks[0].label == "text"
    assert blocks[0].content == "学校 がっこう 학교"
    assert blocks[0].bbox == [10, 20, 200, 60]
    assert blocks[0].order == 1
