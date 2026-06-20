import { useEffect, useState } from "react";

const API = import.meta.env.VITE_API_URL || "";

function ZBar({ z }) {
  const clamped = Math.min(Math.abs(z), 4);
  const pct     = (clamped / 4) * 100;
  const color   = Math.abs(z) > 2
    ? (z > 0 ? "#E24B4A" : "#3B6D11")
    : "#185FA5";
  const left    = z >= 0 ? "50%" : `${50 - pct / 2}%`;
  const width   = `${pct / 2}%`;

  return (
    <div className="z-bar-wrap">
      <div style={{ position:"absolute", left:"50%", width:"1px",
                    height:"100%", background:"#D3D1C7", top:0 }} />
      <div className="z-bar"
           style={{ left, width, background: color }} />
    </div>
  );
}

function SignalBadge({ signal }) {
  if (signal === "long_spread")
    return <span className="signal-badge badge-long">▲ Long spread</span>;
  if (signal === "short_spread")
    return <span className="signal-badge badge-short">▼ Short spread</span>;
  return <span className="signal-badge badge-neutral">No signal</span>;
}

export default function Signals() {
  const [signals, setSignals] = useState([]);
  const [pairs,   setPairs]   = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/signals`).then(r => r.json()),
      fetch(`${API}/api/pairs`).then(r => r.json()),
    ]).then(([s, p]) => {
      setSignals(s);
      setPairs(p);
      setLoading(false);
    });
  }, []);

  if (loading) return <div className="loading">Loading signals…</div>;

  // Merge signals with pair metadata
  const merged = signals.map(s => ({
    ...s,
    ...pairs.find(p => p.pair === s.pair),
  }));

  return (
    <div className="page">
      <div className="page-title">Live Signals</div>
      <div className="page-sub">
        Current z-scores for all cointegrated pairs · entry threshold |z| &gt; 2.0
      </div>

      <div className="signal-grid">
        {merged.map(s => (
          <div className="signal-card" key={s.pair}>
            <div className="signal-pair">{s.pair}</div>
            <div className="signal-tickers">
              β = {s.beta?.toFixed(4)} · half-life {s.halflife_days?.toFixed(0)}d
            </div>
            <ZBar z={s.current_zscore ?? 0} />
            <div style={{ display:"flex", justifyContent:"space-between",
                          alignItems:"center" }}>
              <SignalBadge signal={s.signal} />
              <span style={{ fontSize:13, fontWeight:600,
                             color: Math.abs(s.current_zscore) > 2
                               ? "#E24B4A" : "#2C2C2A" }}>
                z = {s.current_zscore?.toFixed(4)}
              </span>
            </div>
            <div style={{ marginTop:10, display:"grid",
                          gridTemplateColumns:"1fr 1fr", gap:8 }}>
              <div className="metric" style={{ padding:"10px 12px" }}>
                <div className="metric-label">EG p-value</div>
                <div className="metric-value" style={{ fontSize:16 }}>
                  {s.eg_pvalue?.toFixed(3)}
                </div>
              </div>
              <div className="metric" style={{ padding:"10px 12px" }}>
                <div className="metric-label">Entry threshold</div>
                <div className="metric-value" style={{ fontSize:16 }}>
                  ±{s.entry_threshold?.toFixed(1) ?? "2.0"}σ
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}