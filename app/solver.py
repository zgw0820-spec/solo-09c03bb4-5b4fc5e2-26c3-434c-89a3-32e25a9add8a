"""RPC 双程耗时分析核心算法（纯标准库，无第三方依赖）。

问题模型
--------
每台主机 h 有一个本批内固定的未知钟差 o[h]（本地钟读数 = 真实时刻 + o[h]，整数微秒）。
一笔调用：调用方 a、服务方 b，四个时刻
  t1 请求发出（a 钟）、t2 请求到达（b 钟）、t3 应答发出（b 钟）、t4 应答到达（a 钟）。

去程耗时 L_out = (t2 - o[b]) - (t1 - o[a]) = (t2 - t1) - (o[b] - o[a])
回程耗时 L_ret = (t4 - o[a]) - (t3 - o[b]) = (t4 - t3) + (o[b] - o[a])

约束（全部是差分约束 o[v] - o[u] <= w）：
  自洽：L_out >= 0  =>  o[b] - o[a] <= t2 - t1
        L_ret >= 0  =>  o[a] - o[b] <= t4 - t3
  上限 L：L_out <= L  =>  o[a] - o[b] <= L - (t2 - t1)
          L_ret <= L  =>  o[b] - o[a] <= L - (t4 - t3)

隐私与规模
----------
同一对主机之间反复调用只会产生彼此平行的差分约束，平行边取最紧一条后与原约束体系
完全等价。因此把逐笔记录在浏览器端按“有向主机对 (u, v)”聚合成约束摘要（桶键 = 边的
实际方向）：

  * 非负边：去程边 (a, b) 权 t2−t1、回程边 (b, a) 权 t4−t3，同桶取最小权值，记录出处行；
  * 上限边：去程上限边 (b, a)、回程上限边 (a, b)，同桶 delta 取最大值，记录出处行，
    最终边权为 L − delta。

摘要里只含主机名、差值和原表行号，不含调用号与任何原始时刻；约束条数 = 出现过的
有向主机对数（n 台主机至多 n*n 条），不随每对主机的调用数增长。两台主机互相调用时
两个方向各自成条，不会漏约束。自调用（a == b）汇聚为自环边，差值为负即自相矛盾。

每个有向对 (u, v) 的桶内可能同时含两种 leg（互相调用时，对端调用的另一程也落在此
方向），平行边取极值合并。每个有向对最终产出两条同向边：
  非负边 o[v] - o[u] <= nn_w（nn_w 取桶内最小，附胜出约束的原表行号/leg）；
  上限边 o[v] - o[u] <= L - lim_delta（lim_delta 取桶内最大，附出处行号/leg）。

算法：
  * 自洽性 / 上限可行性 / 求一组整数钟差：Bellman-Ford（差分约束经典解法，
    约束全为整数时解天然为整数）；
  * 矛盾定位：Bellman-Ford 前驱指针提取负环，给出带原表行号的约束链；
  * 逐笔可达范围：Floyd-Warshall 全源最短路，o[b]-o[a] 的上下界即最短路；
  * 最小可行整数上限：可行性关于 L 单调，二分查找。
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

MAX_CALLS = 2000
MAX_HOSTS = 16

CSV_FIELDS = ("call_id", "caller", "callee", "req_send", "req_recv", "resp_send", "resp_recv")


@dataclass
class Call:
    call_id: str
    caller: str
    callee: str
    t1: int  # 请求发出（调用方时钟）
    t2: int  # 请求到达（服务方时钟）
    t3: int  # 应答发出（服务方时钟）
    t4: int  # 应答到达（调用方时钟）
    row: int = 0  # 原 CSV 物理行号（1 起，含表头），用于矛盾定位


@dataclass
class PairConstraint:
    """一个有向主机对 (u, v) 的聚合约束摘要。

    非负边：o[v] - o[u] <= nn_w（出处 nn_row，nn_kind 标注去程/回程）；
    上限边（在上限 L 下）：与非负边同向，o[v] - o[u] <= L - lim_delta（出处 lim_row）。
    """

    u: str
    v: str
    nn_w: int
    nn_row: int
    nn_kind: str  # "out" 去程非负 / "ret" 回程非负
    nn_call_id: str | None
    lim_delta: int
    lim_row: int
    lim_kind: str
    lim_call_id: str | None


def parse_csv(text: str) -> list[Call]:
    """解析并校验 CSV 文本，不合法时抛出 ValueError（中文错误信息）。"""
    if not text or not text.strip():
        raise ValueError("CSV 内容为空")
    text = text.lstrip("\ufeff")  # 去掉可能的 BOM
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)  # 容忍前导空行
    text = "\n".join(lines)
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("CSV 缺少表头")
    header = [h.strip() if h else "" for h in reader.fieldnames]
    missing = [f for f in CSV_FIELDS if f not in header]
    if missing:
        raise ValueError(
            "CSV 表头缺少列：{}；需要列：{}".format("、".join(missing), ",".join(CSV_FIELDS))
        )

    calls: list[Call] = []
    for row in reader:
        line = reader.line_num
        if None in row:
            raise ValueError(f"第 {line} 行：列数多于表头，请检查是否多了逗号")
        cleaned = {k.strip(): (v.strip() if v is not None else "") for k, v in row.items()}
        call_id = cleaned["call_id"]
        caller = cleaned["caller"]
        callee = cleaned["callee"]
        if not call_id:
            raise ValueError(f"第 {line} 行：call_id 为空")
        if not caller or not callee:
            raise ValueError(f"第 {line} 行：调用方/服务方主机名为空")
        try:
            t1, t2, t3, t4 = (
                int(cleaned["req_send"]),
                int(cleaned["req_recv"]),
                int(cleaned["resp_send"]),
                int(cleaned["resp_recv"]),
            )
        except ValueError:
            raise ValueError(f"第 {line} 行：四个时刻必须是整数微秒") from None
        calls.append(Call(call_id, caller, callee, t1, t2, t3, t4, line))

    if not calls:
        raise ValueError("CSV 中没有数据行")
    if len(calls) > MAX_CALLS:
        raise ValueError(f"每批最多 {MAX_CALLS} 笔调用，当前 {len(calls)} 笔")
    hosts = {c.caller for c in calls} | {c.callee for c in calls}
    if len(hosts) > MAX_HOSTS:
        raise ValueError(f"每批最多 {MAX_HOSTS} 台主机，当前 {len(hosts)} 台")
    return calls


def build_pair_constraints(
    calls: list[Call],
) -> tuple[list[str], list[PairConstraint]]:
    """把逐笔调用聚合成按有向主机对计的约束摘要（浏览器端与旧 CSV 路径共用同一口径）。

    一笔 a → b 的调用贡献四条边，按方向并入两个桶：
      非负： (a, b, t2-t1, 去程)、(b, a, t4-t3, 回程)
      上限： (b, a, delta=t2-t1, 去程)、(a, b, delta=t4-t3, 回程)
    同一桶内非负权取最小、上限 delta 取最大，并保留胜出约束的原表行号/调用号。
    互相调用的两台主机会在 (a,b)、(b,a) 两个方向同时留条，互不覆盖。
    """
    hosts = sorted({c.caller for c in calls} | {c.callee for c in calls})
    # nn[(u, v)] = 最小 (权值, 行号, 种类, 调用号)；lim[(u, v)] = 最大 delta 同款信息
    nn: dict[tuple[str, str], tuple[int, int, str, str]] = {}
    lim: dict[tuple[str, str], tuple[int, int, str, str]] = {}

    def take_min(key, w, row, kind, call_id):
        old = nn.get(key)
        if old is None or w < old[0]:
            nn[key] = (w, row, kind, call_id)

    def take_max(key, delta, row, kind, call_id):
        old = lim.get(key)
        if old is None or delta > old[0]:
            lim[key] = (delta, row, kind, call_id)

    for c in calls:
        d_out = c.t2 - c.t1
        d_ret = c.t4 - c.t3
        # 桶键 = 边的实际方向。非负：去程边 (a,b)、回程边 (b,a)；
        # 上限：去程上限边 (b,a)、回程上限边 (a,b)。同桶内可能混入两种 leg
        # （互相调用时），平行边取极值后与原约束体系等价。
        take_min((c.caller, c.callee), d_out, c.row, "out", c.call_id)
        take_min((c.callee, c.caller), d_ret, c.row, "ret", c.call_id)
        take_max((c.callee, c.caller), d_out, c.row, "out", c.call_id)
        take_max((c.caller, c.callee), d_ret, c.row, "ret", c.call_id)

    pairs: list[PairConstraint] = []
    for u, v in sorted(set(nn) | set(lim)):
        w, w_row, w_kind, w_call = nn[(u, v)]
        d, d_row, d_kind, d_call = lim[(u, v)]
        pairs.append(
            PairConstraint(
                u=u,
                v=v,
                nn_w=w,
                nn_row=w_row,
                nn_kind=w_kind,
                nn_call_id=w_call,
                lim_delta=d,
                lim_row=d_row,
                lim_kind=d_kind,
                lim_call_id=d_call,
            )
        )
    return hosts, pairs


def validate_summary(hosts: list[str], pairs: list[dict]) -> None:
    """校验浏览器提交的约束摘要，不合法时抛出 ValueError（中文错误信息）。"""
    if not hosts:
        raise ValueError("约束摘要缺少主机清单")
    if len(hosts) > MAX_HOSTS:
        raise ValueError(f"每批最多 {MAX_HOSTS} 台主机，当前 {len(hosts)} 台")
    if len(set(hosts)) != len(hosts):
        raise ValueError("约束摘要的主机清单存在重复主机名")
    name_set = set(hosts)
    if len(pairs) > len(hosts) * len(hosts):
        raise ValueError(
            f"约束条数以有向主机对计上限，最多 {len(hosts) * len(hosts)} 条，当前 {len(pairs)} 条"
        )
    seen: set[tuple[str, str]] = set()
    for k, p in enumerate(pairs):
        u, v = p.get("u"), p.get("v")
        if not isinstance(u, str) or not isinstance(v, str) or u not in name_set or v not in name_set:
            raise ValueError(f"第 {k + 1} 条约束：主机名不在主机清单中")
        if (u, v) in seen:
            raise ValueError(f"第 {k + 1} 条约束：主机对 ({u}, {v}) 重复提交")
        seen.add((u, v))
        for field_name in ("nn_w", "lim_delta"):
            if not isinstance(p.get(field_name), int) or isinstance(p.get(field_name), bool):
                raise ValueError(f"第 {k + 1} 条约束：{field_name} 必须是整数")
        for field_name in ("nn_row", "lim_row"):
            val = p.get(field_name)
            if not isinstance(val, int) or isinstance(val, bool) or val < 1:
                raise ValueError(f"第 {k + 1} 条约束：{field_name} 必须是正整数行号")


def bellman_ford(n: int, edges: list[tuple]) -> list[int] | None:
    """差分约束求解：边 (u, v, w) 表示 x[v] - x[u] <= w。

    返回一组整数解（各分量 <= 0），存在负环（不可行）时返回 None。
    所有距离初始为 0，等价于从超级源点引 0 权边，非连通图同样适用。
    """
    dist = [0] * n
    for _ in range(n - 1):
        updated = False
        for u, v, w, *_ in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            break
    for u, v, w, *_ in edges:
        if dist[u] + w < dist[v]:
            return None
    return dist


def find_negative_cycle(n: int, edges: list[tuple]) -> tuple[list[tuple], int] | None:
    """定位负环：返回 (环上的边列表, 环总权值)，无负环返回 None。边可带第 4 元素作为说明。"""
    dist = [0] * n
    pred = [-1] * n  # pred[v] = 使 v 松弛的边的下标
    last = -1
    for _ in range(n):
        last = -1
        for k, edge in enumerate(edges):
            u, v, w = edge[0], edge[1], edge[2]
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = k
                last = v
    if last == -1:
        return None
    y = last
    for _ in range(n):
        y = edges[pred[y]][0]
    steps: list[tuple] = []
    seen: dict[int, int] = {}
    cur = y
    while cur not in seen:
        seen[cur] = len(steps)
        k = pred[cur]
        steps.append(edges[k])
        cur = edges[k][0]
    steps = steps[seen[cur]:]
    total = sum(e[2] for e in steps)
    return steps, total


def floyd_warshall(n: int, edges: list[tuple]) -> list[list[float]]:
    """全源最短路（调用前需保证无负环）。d[i][j] 即 x[j] - x[i] 的最紧上界。"""
    inf = float("inf")
    d = [[inf] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = 0
    for u, v, w, *_ in edges:
        if w < d[u][v]:
            d[u][v] = w
    for k in range(n):
        dk = d[k]
        for i in range(n):
            di = d[i]
            ik = di[k]
            if ik == inf:
                continue
            for j in range(n):
                kj = dk[j]
                if kj != inf and ik + kj < di[j]:
                    di[j] = ik + kj
    return d


def _make_edges(
    pairs: list[PairConstraint] | list[dict],
    idx: dict[str, int],
    limit: int,
    include_limit: bool,
) -> list[tuple]:
    """由约束摘要构造 Bellman-Ford 边 (u, v, w, meta)。

    自洽判定只用非负边（include_limit=False）；上限可行性再叠加上限边。
    摘要既可以是内部 PairConstraint，也可以是 API 进来的 dict。
    """
    edges: list[tuple] = []
    for p in pairs:
        u, v = p["u"] if isinstance(p, dict) else p.u, p["v"] if isinstance(p, dict) else p.v
        i, j = idx[u], idx[v]
        nn_w = p["nn_w"] if isinstance(p, dict) else p.nn_w
        nn_row = p["nn_row"] if isinstance(p, dict) else p.nn_row
        nn_kind = p.get("nn_kind", "out") if isinstance(p, dict) else p.nn_kind
        nn_call = p.get("nn_call_id") if isinstance(p, dict) else p.nn_call_id
        edges.append(
            (
                i,
                j,
                nn_w,
                {"kind": "nonneg", "leg": nn_kind, "row": nn_row, "call_id": nn_call,
                 "u": u, "v": v},
            )
        )
        if include_limit:
            lim_delta = p["lim_delta"] if isinstance(p, dict) else p.lim_delta
            lim_row = p["lim_row"] if isinstance(p, dict) else p.lim_row
            lim_kind = p.get("lim_kind", "out") if isinstance(p, dict) else p.lim_kind
            lim_call = p.get("lim_call_id") if isinstance(p, dict) else p.lim_call_id
            # 上限边与非负边同向：o[v] - o[u] <= L - lim_delta
            edges.append(
                (
                    i,
                    j,
                    limit - lim_delta,
                    {"kind": "limit", "leg": lim_kind, "row": lim_row, "call_id": lim_call,
                     "u": u, "v": v},
                )
            )
    return edges


def _offsets_dict(hosts: list[str], sol: list[int]) -> dict[str, int]:
    base = min(sol)  # 整体平移不影响耗时，归零便于阅读
    return {h: sol[i] - base for i, h in enumerate(hosts)}


def analyze_constraints(hosts: list[str], pairs: list, limit_us: int) -> dict:
    """约束摘要主分析入口（不含逐笔数据：逐笔范围/耗时由持有原表的一方计算）。"""
    n = len(hosts)
    idx = {h: i for i, h in enumerate(hosts)}

    # 1) 自洽性：只看非负边
    nn_edges = _make_edges(pairs, idx, limit_us, include_limit=False)
    cycle = find_negative_cycle(n, nn_edges)
    if cycle is not None:
        steps, total = cycle
        return {
            "consistent": False,
            "hosts": hosts,
            "limit_us": limit_us,
            "contradiction": {
                "total_us": total,
                "steps": [
                    {
                        "kind": e[3]["kind"],
                        "leg": e[3]["leg"],
                        "u": hosts[e[0]],
                        "v": hosts[e[1]],
                        "w": e[2],
                        "row": e[3]["row"],
                        "call_id": e[3]["call_id"],
                    }
                    for e in steps
                ],
            },
        }

    # 2) 全源最短路：o[v] - o[u] 的最紧上界（只输出有限项，供逐笔范围计算）
    dist = floyd_warshall(n, nn_edges)
    bounds = [
        {"u": hosts[i], "v": hosts[j], "upper": int(dist[i][j])}
        for i in range(n)
        for j in range(n)
        if i != j and dist[i][j] != float("inf")
    ]

    def solve_with(limit: int) -> list[int] | None:
        return bellman_ford(n, _make_edges(pairs, idx, limit, include_limit=True))

    # 3) 当前上限可行性
    sol = solve_with(limit_us)
    result: dict = {
        "consistent": True,
        "hosts": hosts,
        "limit_us": limit_us,
        "bounds": bounds,
    }
    if sol is not None:
        result["feasible_with_limit"] = True
        result["min_limit_us"] = None
        result["offsets"] = _offsets_dict(hosts, sol)
        result["offsets_note"] = "满足当前上限的一组钟差"
        return result

    # 4) 自洽但无法同时达标：二分最小可行整数上限
    lo, hi = -1, 1  # lo 不可行（至少一条约束时 -1 必不可行），倍增找可行的 hi
    while solve_with(hi) is None:
        hi *= 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if solve_with(mid) is not None:
            hi = mid
        else:
            lo = mid
    sol = solve_with(hi)
    result["feasible_with_limit"] = False
    result["min_limit_us"] = hi
    result["offsets"] = _offsets_dict(hosts, sol)
    result["offsets_note"] = "满足最小可行上限的一组钟差"
    return result


def _legacy_step_text(step: dict) -> str:
    """把结构化矛盾链翻译回旧版 /api/analyze 的中文描述（含调用号）。"""
    leg = "去程非负" if step["leg"] == "out" else "回程非负"
    return f"o[{step['v']}] − o[{step['u']}] ≤ {step['w']}（调用 {step['call_id']} {leg}）"


def _latencies(calls: list[Call], offsets: dict[str, int]) -> list[dict]:
    """按给定钟差计算每笔调用的去程/回程耗时。"""
    out = []
    for c in calls:
        d = offsets[c.callee] - offsets[c.caller]
        out.append(
            {
                "call_id": c.call_id,
                "caller": c.caller,
                "callee": c.callee,
                "out_us": (c.t2 - c.t1) - d,
                "ret_us": (c.t4 - c.t3) + d,
            }
        )
    return out


def _call_ranges(calls: list[Call], bounds_map: dict[tuple[str, str], int], limit_us: int) -> list[dict]:
    """按全源最短路界推出逐笔去/回程可达范围（与旧版同口径）。"""
    call_ranges = []
    for c in calls:
        d_ab = bounds_map[(c.caller, c.callee)]  # o[b] - o[a] 的最大值
        d_ba = bounds_map[(c.callee, c.caller)]  # o[a] - o[b] 的最大值
        d_out, d_ret = c.t2 - c.t1, c.t4 - c.t3
        out_min = d_out - d_ab
        out_max = d_out + d_ba
        ret_min = d_ret - d_ba
        ret_max = d_ret + d_ab
        call_ranges.append(
            {
                "call_id": c.call_id,
                "caller": c.caller,
                "callee": c.callee,
                "out_min": out_min,
                "out_max": out_max,
                "ret_min": ret_min,
                "ret_max": ret_max,
                "must_exceed": out_min > limit_us or ret_min > limit_us,
            }
        )
    return call_ranges


def analyze(calls: list[Call], limit_us: int) -> dict:
    """旧版 CSV 全量分析入口：在约束摘要之上补算逐笔范围与耗时，响应结构保持不变。"""
    hosts, pairs = build_pair_constraints(calls)
    result = analyze_constraints(hosts, pairs, limit_us)

    if not result["consistent"]:
        result["contradiction"]["steps"] = [
            _legacy_step_text(s) for s in result["contradiction"]["steps"]
        ]
        return result

    bounds_map = {(b["u"], b["v"]): b["upper"] for b in result["bounds"]}
    # Floyd 界里省略了对角项；同主机钟差恒为 0，o[h] - o[h] 的上界为 0（自调用场景需要）
    for h in hosts:
        bounds_map[(h, h)] = 0
    result["calls"] = _call_ranges(calls, bounds_map, limit_us)
    note = result.pop("offsets_note")
    result["assignment"] = _latencies(calls, result["offsets"])
    result["assignment_note"] = note
    # 保持旧版字段顺序观感：bounds 是摘要路径才需要的内部量，旧响应里去掉
    result.pop("bounds")
    return result
