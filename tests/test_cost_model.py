import torch

from hardware_model.cost_model import ScampCostModel, compare_reports, route_depth, shortest_route_depth


def test_known_qat_counts_and_shape():
    conv = torch.zeros(16, 1, 4, 4)
    flat = conv.view(-1)
    flat[:128] = 1
    flat[128:128 + 89] = -1
    fc = torch.zeros(10, 4096)
    fc_flat = fc.view(-1)
    fc_flat[:5404] = 1
    fc_flat[5404:5404 + 6709] = -1
    report = ScampCostModel().analyze_ternary("synthetic-count-check", conv, fc)
    assert report.positive_terms == 529692
    assert report.negative_terms == 371253
    assert report.active_terms == 900945
    assert report.skipped_terms == 188591
    assert report.dense_terms == 1089536
    assert report.total_offsets == 16


def test_group_zero_removes_schedule_tap():
    conv = torch.ones(16, 1, 4, 4)
    fc = torch.ones(10, 4096)
    base = ScampCostModel().analyze_ternary("base", conv, fc)
    conv[:, :, 1, 2] = 0
    sparse = ScampCostModel().analyze_ternary("sparse", conv, fc)
    assert sparse.retained_offsets == base.retained_offsets - 1
    assert sparse.active_terms == base.active_terms - 16 * 64 * 64
    comparison = compare_reports(sparse, base)
    assert comparison["improvement_percent"]["compute"] > 0
    assert comparison["improvement_percent"]["latency"] > 0


def test_route_depth_is_explicit():
    assert route_depth([(0, 0), (0, 1), (1, 1)]) == 2
    assert shortest_route_depth([(0, 0), (0, 1), (1, 1)]) == 2
    assert shortest_route_depth([(0, 0), (3, 3)]) == 6
