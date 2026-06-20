import { useEffect, useState } from "react";

const API = import.meta.env.VITE_API_URL || "";

export default function Trades() {
  const [trades,  setTrades]  = useState([]);
  const [filter,  setFilter]  = useState("all");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API}/api/trades`)
      .then(r => r.json())
      .then(d => { setTrades(d); setLoading(false); });
  }, []);

  if (loading) return <div className="loading">Loading trades…</div>;

  const pairs   = ["all", ...new Set(trades.map(t => t.pair))];
  const filtered = filter === "all"
    ? trades
    : trades.filter(t => t.pair === filter);

  const totalPnl  = filtered.reduce((s, t) => s + t.net_pnl, 0);
  const wins      = filtered.filter(t => t.net_pnl > 0).length;
  const winRate   = filtered.length ? (wins / filtered.length * 100) : 0;

  const fmt = d => d?.slice(0, 10) ?? "—";
  const fmtPnl = v => (v >= 0 ? "+" : "") + "$" + Math.abs(v).toLocaleString(
    undefined, { minimumFractionDigits:2, maximumFractionDigits:2 });

  return (
    <div className="page">
      <div className="page-title">Trade Log</div>
      <div className="page-sub">
        Full backtest history · all completed round trips
      </div>

      {/* Summary row */}
      <div className="metrics-grid" style={{ marginBottom:"1rem" }}>
        <div className="metric">
          <div className="metric-label">Showing</div>
          <div className="metric-value" style={{ fontSize:18 }}>
            {filtered.length} trades
          </div>
        </div>
        <div className="metric">
          <div className="metric-label">Net PnL</div>
          <div className={`metric-value ${totalPnl >= 0 ? "pos" : "neg"}`}
               style={{ fontSize:18 }}>
            {fmtPnl(totalPnl)}
          </div>
        </div>
        <div className="metric">
          <div className="metric-label">Win Rate</div>
          <div className="metric-value" style={{ fontSize:18 }}>
            {winRate.toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Filter tabs */}
      <div style={{ display:"flex", gap:6, marginBottom:"1rem" }}>
        {pairs.map(p => (
          <button
            key={p}
            onClick={() => setFilter(p)}
            style={{
              fontSize:12, padding:"5px 12px",
              borderRadius:6, border:"1px solid #E8E6DF",
              background: filter === p ? "#E6F1FB" : "#fff",
              color: filter === p ? "#185FA5" : "#5F5E5A",
              cursor:"pointer", fontWeight: filter === p ? 500 : 400,
            }}
          >
            {p}
          </button>
        ))}
      </div>

      {/* Table */}
      <div className="card" style={{ padding:0, overflow:"hidden" }}>
        <div style={{ overflowX:"auto" }}>
          <table>
            <thead>
              <tr>
                <th>Pair</th>
                <th>Direction</th>
                <th>Entry date</th>
                <th>Exit date</th>
                <th>Entry z</th>
                <th>Exit z</th>
                <th>Hold (days)</th>
                <th>Exit type</th>
                <th>Net PnL</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((t, i) => (
                <tr key={i}>
                  <td style={{ fontWeight:500 }}>{t.pair}</td>
                  <td>
                    <span style={{
                      fontSize:11, padding:"2px 8px", borderRadius:20,
                      background: t.direction === "long_spread"
                        ? "#EAF3DE" : "#FCEBEB",
                      color: t.direction === "long_spread"
                        ? "#27500A" : "#791F1F",
                    }}>
                      {t.direction === "long_spread" ? "▲ Long" : "▼ Short"}
                    </span>
                  </td>
                  <td style={{ color:"#5F5E5A" }}>{fmt(t.entry_date)}</td>
                  <td style={{ color:"#5F5E5A" }}>{fmt(t.exit_date)}</td>
                  <td style={{ fontFamily:"monospace" }}>
                    {t.entry_z?.toFixed(3)}
                  </td>
                  <td style={{ fontFamily:"monospace" }}>
                    {t.exit_z?.toFixed(3)}
                  </td>
                  <td style={{ textAlign:"center" }}>{t.holding_days}</td>
                  <td>
                    <span style={{
                      fontSize:11, padding:"2px 8px", borderRadius:20,
                      background: t.exit_type === "stop_loss"
                        ? "#FAEEDA" : "#F1EFE8",
                      color: t.exit_type === "stop_loss"
                        ? "#633806" : "#5F5E5A",
                    }}>
                      {t.exit_type === "stop_loss" ? "Stop loss" : "Reversion"}
                    </span>
                  </td>
                  <td className={t.net_pnl >= 0 ? "pnl-pos" : "pnl-neg"}>
                    {fmtPnl(t.net_pnl)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}