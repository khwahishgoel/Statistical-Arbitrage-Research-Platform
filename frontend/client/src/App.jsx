import { useState } from "react";
import Signals  from "./pages/Signals";
import Backtest from "./pages/Backtest";
import Trades   from "./pages/Trades";

const PAGES = ["Signals", "Backtest", "Trades"];

export default function App() {
  const [page, setPage] = useState("Signals");

  return (
    <>
      <nav className="nav">
        <span className="nav-brand">Stat Arb Platform</span>
        <div className="nav-links">
          {PAGES.map(p => (
            <button
              key={p}
              className={`nav-link${page === p ? " active" : ""}`}
              onClick={() => setPage(p)}
            >
              {p}
            </button>
          ))}
        </div>
      </nav>

      {page === "Signals"  && <Signals />}
      {page === "Backtest" && <Backtest />}
      {page === "Trades"   && <Trades />}
    </>
  );
}