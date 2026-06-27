import torch


def test_mask2former_forward_shape():
    """Test Mask2Former produces expected output shapes."""
    from transformers import Mask2FormerForUniversalSegmentation

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-base-coco-panoptic",
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    batch = 2
    pixel_values = torch.randn(batch, 3, 384, 384)
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    assert outputs.masks_queries_logits is not None, "masks_queries_logits should not be None"
    assert outputs.class_queries_logits is not None, "class_queries_logits should not be None"
    assert outputs.class_queries_logits.shape[0] == batch, "batch dim mismatch"
    assert outputs.class_queries_logits.shape[2] == model.config.num_labels + 1, (
        f"expected {model.config.num_labels + 1} classes "
        f"(num_labels + no-object), got {outputs.class_queries_logits.shape[2]}"
    )


def test_mask2former_config():
    """Test model config reflects num_labels."""
    from transformers import Mask2FormerForUniversalSegmentation

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-base-coco-panoptic",
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    assert model.config.num_labels == 2
