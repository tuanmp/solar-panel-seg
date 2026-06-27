import pytest
import torch
from solar_seg.evaluation.metrics import panoptic_quality, mean_iou


def test_panoptic_quality_perfect():
    sem = torch.zeros((100, 100), dtype=torch.long)
    inst = torch.zeros((100, 100), dtype=torch.long)
    inst[10:40, 10:40] = 1
    sem[10:40, 10:40] = 1

    result = panoptic_quality(sem, inst, sem, inst, num_classes=2)
    assert result["pq"] == pytest.approx(1.0, abs=0.01)


def test_panoptic_quality_no_match():
    sem1 = torch.zeros((100, 100), dtype=torch.long)
    inst1 = torch.zeros((100, 100), dtype=torch.long)
    sem2 = torch.ones((100, 100), dtype=torch.long)
    inst2 = torch.ones((100, 100), dtype=torch.long)

    result = panoptic_quality(sem1, inst1, sem2, inst2, num_classes=2)
    assert result["pq"] == 0.0


def test_mean_iou_perfect():
    pred = torch.zeros((100, 100), dtype=torch.long)
    target = torch.zeros((100, 100), dtype=torch.long)
    assert mean_iou(pred, target, num_classes=2) == pytest.approx(1.0, abs=0.01)


def test_mean_iou_partial():
    pred = torch.zeros((100, 100), dtype=torch.long)
    pred[:50] = 1
    target = torch.zeros((100, 100), dtype=torch.long)
    target[50:] = 1
    result = mean_iou(pred, target, num_classes=2)
    assert 0.0 < result < 1.0
