import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from "recharts";

const API = import.meta.env.VITE_API_URL || "";

function MetricCard({ label, value, sub, color }) {
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className={`metric-value ${color || ""}`}>{value}</div>
      {sub && <div style={{ fontSize:11, color:"#888780", marginTop:2 }}>{sub}</div>}
    </div>
  );
}

export default function Backtest() {
  const [bt,      setBt]      = useState(null);
  const [hmm,     setHmm]     = useState(null);
  const [trades,  setTrades]  = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/backtest`).then(r => r.json()),
      fetch(`${API}/api/hmm`).then(r => r.json()),
      fetch(`${API}/api/trades`).then(r => r.json()),
    ]).then(([b, h, t]) => {
      setBt(b);
      setHmm(h);
      setTrades(t);
      setLoading(false);
    });
  }, []);

  if (loading) return <div className="loading">Loading backtest…</div>;

  // Build cumulative PnL series from trades
  const sorted = [...trades].sort(
    (a, b) => new Date(a.exit_date) - new Date(b.exit_date)
  );
  let cum = 0;
  const chartData = sorted.map(t => {
    cum += t.net_pnl;
    return {
      date: t.exit_date?.slice(0, 10),
      pnl:  Math.round(cum),
      pair: t.pair,
    };
  });

  // Per-pair cumulative
  const pairs = [...new Set(trades.map(t => t.pair))];
  const pairColors = {
    "CVX/XOM": "#E24B4A",
    "GS/MS":   "#185FA5",
    "XLE/XOM": "#3B6D11",
  };

  const pairData = {};
  pairs.forEach(p => {
    let c = 0;
    pairData[p] = sorted
      .filter(t => t.pair === p)
      .map(t => {
        c += t.net_pnl;
        return { date: t.exit_date?.slice(0, 10), [p]: Math.round(c) };
      });
  });

  // Merge pair data by date
  const allDates = [...new Set(sorted.map(t => t.exit_date?.slice(0, 10)))]
    .sort();
  const pairChartData = allDates.map(date => {
    const row = { date };
    pairs.forEach(p => {
      const match = pairData[p].find(d => d.date === date);
      if (match) row[p] = match[p];
    });
    return row;
  });
  // Forward-fill
  const lastVals = {};
  pairChartData.forEach(row => {
    pairs.forEach(p => {
      if (row[p] !== undefined) lastVals[p] = row[p];
      else if (lastVals[p] !== undefined) row[p] = lastVals[p];
    });
  });

  const fmt = v => `$${v?.toLocaleString()}`;

  return (
    <div className="page">
      <div className="page-title">Backtest Results</div>
      <div className="page-sub">
        4.5 year backtest · 3 pairs · $10k per leg · 10bps transaction costs
      </div>

      {/* Metrics — baseline */}
      <div className="card">
        <div className="card-title">Baseline strategy</div>
        <div className="metrics-grid">
          <MetricCard label="Total PnL"     value={fmt(bt?.total_pnl)}          color="pos" />
          <MetricCard label="Sharpe Ratio"  value={bt?.sharpe_ratio?.toFixed(2)} color="blue" />
          <MetricCard label="Win Rate"      value={`${bt?.win_rate?.toFixed(1)}%`} />
          <MetricCard label="Max Drawdown"  value={fmt(bt?.max_drawdown)}        color="neg" />
          <MetricCard label="Profit Factor" value={bt?.profit_factor?.toFixed(2)} />
          <MetricCard label="Total Trades"  value={bt?.total_trades} />
          <MetricCard label="Avg Hold"      value={`${bt?.avg_hold_days?.toFixed(1)}d`} />
          <MetricCard label="Ann. Return"   value={`${bt?.annualised_return?.toFixed(1)}%`} />
        </div>
      </div>

      {/* HMM comparison */}
      {hmm?.baseline && (
        <div className="card">
          <div className="card-title">HMM regime filter impact</div>
          <div className="compare-grid">
            <div>
              <div style={{ fontSize:12, color:"#888780", marginBottom:8 }}>
                Baseline
              </div>
              <div className="metrics-grid" style={{ gridTemplateColumns:"1fr 1fr" }}>
                <MetricCard label="Sharpe"   value={hmm.baseline.sharpe_ratio?.toFixed(2)} />
                <MetricCard label="Win Rate" value={`${hmm.baseline.win_rate?.toFixed(1)}%`} />
                <MetricCard label="PnL"      value={fmt(hmm.baseline.total_pnl)} />
                <MetricCard label="Trades"   value={hmm.baseline.total_trades} />
              </div>
            </div>
            <div>
              <div style={{ fontSize:12, color:"#3B6D11", marginBottom:8,
                            fontWeight:500 }}>
                HMM-filtered ↑
              </div>
              <div className="metrics-grid" style={{ gridTemplateColumns:"1fr 1fr" }}>
                <MetricCard label="Sharpe"   value={hmm.filtered.sharpe_ratio?.toFixed(2)} color="pos" />
                <MetricCard label="Win Rate" value={`${hmm.filtered.win_rate?.toFixed(1)}%`} color="pos" />
                <MetricCard label="PnL"      value={fmt(hmm.filtered.total_pnl)} color="pos" />
                <MetricCard label="Trades"   value={hmm.filtered.total_trades} />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Portfolio equity curve */}
      <div className="card">
        <div className="card-title">Portfolio cumulative PnL</div>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={chartData}
                     margin={{ top:4, right:16, bottom:0, left:16 }}>
            <CartesianGrid stroke="#F1EFE8" vertical={false} />
            <XAxis dataKey="date" tick={{ fontSize:11 }}
                   tickFormatter={d => d?.slice(0,7)} interval={30} />
            <YAxis tick={{ fontSize:11 }}
                   tickFormatter={v => `$${(v/1000).toFixed(0)}k`} />
            <Tooltip formatter={v => [`$${v.toLocaleString()}`, "PnL"]}
                     labelStyle={{ fontSize:11 }} />
            <ReferenceLine y={0} stroke="#D3D1C7" />
            <Line type="monotone" dataKey="pnl" stroke="#185FA5"
                  dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Per-pair breakdown */}
      <div className="card">
        <div className="card-title">Per-pair cumulative PnL</div>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={pairChartData}
                     margin={{ top:4, right:16, bottom:0, left:16 }}>
            <CartesianGrid stroke="#F1EFE8" vertical={false} />
            <XAxis dataKey="date" tick={{ fontSize:11 }}
                   tickFormatter={d => d?.slice(0,7)} interval={30} />
            <YAxis tick={{ fontSize:11 }}
                   tickFormatter={v => `$${(v/1000).toFixed(0)}k`} />
            <Tooltip formatter={(v, name) => [`$${v?.toLocaleString()}`, name]}
                     labelStyle={{ fontSize:11 }} />
            <ReferenceLine y={0} stroke="#D3D1C7" />
            <Legend wrapperStyle={{ fontSize:12 }} />
            {pairs.map(p => (
              <Line key={p} type="monotone" dataKey={p}
                    stroke={pairColors[p] || "#888780"}
                    dot={false} strokeWidth={1.5}
                    connectNulls />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}