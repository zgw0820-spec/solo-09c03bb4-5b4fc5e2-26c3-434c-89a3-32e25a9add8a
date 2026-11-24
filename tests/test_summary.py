"""约束摘要路径测试：按主机对聚合、互相调用不漏约束、矛盾行号定位、规模上限。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import pytest  # noqa: E402

import solver  # noqa: E402

SAMPLE_CSV = """\
call_id,caller,callee,req_send,req_recv,resp_send,resp_recv
c1,web1,app1,1000000,1000620,1000700,1000350
c2,app1,db1,1000650,999930,1000000,1000920
c3,web1,db1,1000500,1000400,1000500,1001050
"""

# 同一主机对 web1→app1 反复 3 笔 + 反向 app1→web1 1 笔（互相调用），全部差值为正、整体自洽
REPEATED_PAIR_CSV = """\
call_id,caller,callee,req_send,req_recv,resp_send,resp_recv
dup,web1,app1,1000000,1000600,1000700,1001300
dup,web1,app1,1001000,1001550,1001650,1002250
dup,web1,app1,1002000,1002610,1002700,1003290
back,app1,web1,1003000,1003400,1003500,1003800
"""

# 矛盾样例：三笔调用号完全相同，每笔单独自洽（d_out+d_ret>=0），
# 但三条去程边组成的三角环总权为负 → 矛盾链横跨第 2/3/4 行，只能靠行号定位
DUP_ID_CONTRADICTION_CSV = """\
call_id,caller,callee,req_send,req_recv,resp_send,resp_recv
x1,web1,app1,1000000,999950,1000000,1000060
x1,app1,db1,2000000,1999940,2000000,2000070
x1,db1,web1,3000000,2999970,3000000,3000040
"""


def test_repeated_pair_aggregates_by_directed_pair():
    calls = solver.parse_csv(REPEATED_PAIR_CSV)
    hosts, pairs = solver.build_pair_constraints(calls)
    # 4 笔调用只产生 2 个有向主机对：(web1,app1) 与 (app1,web1)
    assert hosts == ["app1", "web1"]
    assert len(pairs) == 2
    by_key = {(p.u, p.v): p for p in pairs}
    fwd = by_key[("web1", "app1")]
    rev = by_key[("app1", "web1")]
    # (web1,app1) 桶：非负边 = 三笔去程 d_out(600/550/610) 与 back 回程 d_ret(300) 取最小
    assert fwd.nn_w == 300 and fwd.nn_row == 5 and fwd.nn_kind == "ret"
    # 上限边 = 三笔回程上限 delta(600/600/590) 与 back 去程上限 delta(400) 取最大
    assert fwd.lim_delta == 600 and fwd.lim_row in (2, 3) and fwd.lim_kind == "ret"
    # (app1,web1) 桶：非负边 = 三笔回程 d_ret(600/600/590) 与 back 去程 d_out(400) 取最小
    assert rev.nn_w == 400 and rev.nn_row == 5 and rev.nn_kind == "out"
    # 上限边 = 三笔去程上限 delta(600/550/610) 与 back 回程上限 delta(300) 取最大
    assert rev.lim_delta == 610 and rev.lim_row == 4 and rev.lim_kind == "out"
    # 反向调用 back 也必须留约束，不能被同向记录吞掉
    back_pairs = [p for p in pairs if p.u == "app1" and p.v == "web1"]
    assert len(back_pairs) == 1


def test_summary_equivalent_to_full_analysis_on_sample():
    calls = solver.parse_csv(SAMPLE_CSV)
    hosts, pairs = solver.build_pair_constraints(calls)
    for limit in (0, 200, 224, 225, 250, 1000):
        full = solver.analyze(calls, limit)
        summary = solver.analyze_constraints(hosts, pairs, limit)
        assert summary["consistent"] == full["consistent"]
        assert summary["feasible_with_limit"] == full["feasible_with_limit"]
        assert summary["min_limit_us"] == full["min_limit_us"]
        assert summary["offsets"] == full["offsets"]
        # 逐笔范围与摘要 bounds 推出的界一致
        bmap = {(b["u"], b["v"]): b["upper"] for b in summary["bounds"]}
        for h in hosts:
            bmap[(h, h)] = 0
        for c, rng in zip(calls, full["calls"]):
            d_ab, d_ba = bmap[(c.caller, c.callee)], bmap[(c.callee, c.caller)]
            assert rng["out_min"] == (c.t2 - c.t1) - d_ab
            assert rng["out_max"] == (c.t2 - c.t1) + d_ba
            assert rng["ret_min"] == (c.t4 - c.t3) - d_ba
            assert rng["ret_max"] == (c.t4 - c.t3) + d_ab


def test_repeated_pair_summary_result_matches_full():
    calls = solver.parse_csv(REPEATED_PAIR_CSV)
    hosts, pairs = solver.build_pair_constraints(calls)
    full = solver.analyze(calls, 400)
    summary = solver.analyze_constraints(hosts, pairs, 400)
    assert summary["consistent"] is full["consistent"]
    assert summary["feasible_with_limit"] == full["feasible_with_limit"]
    assert summary["offsets"] == full["offsets"]
    # 摘要给出的钟差下，逐笔耗时在浏览器侧重算必须全部落在 [0, min_limit 或 400]
    for c in calls:
        d = summary["offsets"][c.callee] - summary["offsets"][c.caller]
        out = (c.t2 - c.t1) - d
        ret = (c.t4 - c.t3) + d
        assert out >= 0 and ret >= 0


def test_constraint_count_bounded_by_host_pairs_not_call_count():
    # 2000 笔、全部同一有向主机对 → 摘要只有 2 个有向对（去程/回程各一）
    header = "call_id,caller,callee,req_send,req_recv,resp_send,resp_recv\n"
    rows = "".join(f"k{i},a,b,{1000 + i},{1100 + i},{1200 + i},{1250 + i}\n" for i in range(2000))
    calls = solver.parse_csv(header + rows)
    hosts, pairs = solver.build_pair_constraints(calls)
    assert len(calls) == 2000 and len(hosts) == 2
    assert len(pairs) == 2
    # 16 台主机满配：有向对至多 16*16 = 256 条
    rows16 = "".join(
        f"k{i},h{i % 16},h{(i + 1) % 16},{i},{i + 10},{i + 20},{i + 30}\n" for i in range(2000)
    )
    calls16 = solver.parse_csv(header + rows16)
    hosts16, pairs16 = solver.build_pair_constraints(calls16)
    assert len(hosts16) == 16
    assert len(pairs16) <= 16 * 16


def test_contradiction_locates_original_rows_with_duplicate_call_id():
    calls = solver.parse_csv(DUP_ID_CONTRADICTION_CSV)
    hosts, pairs = solver.build_pair_constraints(calls)
    r = solver.analyze_constraints(hosts, pairs, 1000)
    assert r["consistent"] is False
    rows = sorted(s["row"] for s in r["contradiction"]["steps"])
    # 三条记录的物理行号是 2/3/4；调用号同为 x1，仍必须能按行号区分
    assert rows == [2, 3, 4]
    assert all(s["call_id"] == "x1" for s in r["contradiction"]["steps"])
    assert r["contradiction"]["total_us"] == -140
    # 旧 /api/analyze 路径的中文链同样保留行号信息（经调用号描述）
    legacy = solver.analyze(calls, 1000)
    assert legacy["consistent"] is False
    assert any("x1" in s for s in legacy["contradiction"]["steps"])


def test_single_call_contradiction_row_number():
    calls = solver.parse_csv(
        "call_id,caller,callee,req_send,req_recv,resp_send,resp_recv\n"
        "x1,web1,app1,1000000,1000100,1000400,1000200\n"
    )
    hosts, pairs = solver.build_pair_constraints(calls)
    r = solver.analyze_constraints(hosts, pairs, 1000)
    assert r["consistent"] is False
    assert r["contradiction"]["total_us"] == -100
    assert {s["row"] for s in r["contradiction"]["steps"]} == {2}


def test_validate_summary_rejects_over_limit_and_bad_names():
    hosts = [f"h{i}" for i in range(16)]
    pairs = [{"u": "h0", "v": f"h{i}", "nn_w": 1, "nn_row": 2,
              "lim_delta": 1, "lim_row": 2} for i in range(257)]
    with pytest.raises(ValueError, match="主机对计上限"):
        solver.validate_summary(hosts, pairs)
    with pytest.raises(ValueError, match="主机清单"):
        solver.validate_summary(hosts, [{**pairs[0], "u": "ghost"}])
    with pytest.raises(ValueError, match="重复提交"):
        solver.validate_summary(["a", "b"], [
            {"u": "a", "v": "b", "nn_w": 1, "nn_row": 1, "lim_delta": 1, "lim_row": 1},
            {"u": "a", "v": "b", "nn_w": 2, "nn_row": 2, "lim_delta": 2, "lim_row": 2},
        ])
    with pytest.raises(ValueError, match="正整数行号"):
        solver.validate_summary(["a", "b"], [
            {"u": "a", "v": "b", "nn_w": 1, "nn_row": 0, "lim_delta": 1, "lim_row": 1},
        ])


def test_validate_summary_accepts_max_256_pairs():
    hosts = [f"h{i}" for i in range(16)]
    pairs = [
        {"u": u, "v": v, "nn_w": 10, "nn_row": 2, "lim_delta": 5, "lim_row": 2}
        for u in hosts for v in hosts
    ]
    solver.validate_summary(hosts, pairs)  # 不抛异常即可
