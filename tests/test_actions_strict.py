from cartoweave.data.api import make_solvepack_from_data_defaults


def test_action_sequence_strict_properties():
    pack = make_solvepack_from_data_defaults("configs/data.yaml")
    assert pack.action_num == len(pack.actions)
    assert pack.behaviors == []
    times = [a.t for a in pack.actions]
    assert all(0.0 < t < 1.0 for t in times)
    assert all(t2 > t1 for t1, t2 in zip(times, times[1:]))

    by_label: dict[int, list] = {}
    for act in pack.actions:
        by_label.setdefault(act.id, []).append(act)
    for seq in by_label.values():
        assert seq[0].type == "appear"
        if len(seq) > 1:
            for s in seq[1:-1]:
                assert s.type == "mutate"
            assert seq[-1].type in {"mutate", "disappear"}

    for act in pack.actions:
        if act.type == "appear":
            assert act.WH_to == pack.labels0[act.id].WH
            assert act.kind_to == "rectangular"
        elif act.type == "mutate":
            assert act.WH_to is not None
