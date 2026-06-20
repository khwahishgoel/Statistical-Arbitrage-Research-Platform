const express = require("express");
const cors    = require("cors");
const { MongoClient } = require("mongodb");
require("dotenv").config();

const app  = express();
const PORT = process.env.PORT || 4000;

// ── MongoDB ───────────────────────────────────────────────────────────────────
let db;
async function connectDB() {
  const client = new MongoClient(process.env.MONGO_URI, {
    tls: true,
    tlsCAFile: process.env.TLS_CA_FILE || undefined,
  });
  await client.connect();
  db = client.db("stat_arb");
  console.log("Connected to MongoDB Atlas");
}

// ── Middleware ────────────────────────────────────────────────────────────────
app.use(cors({ origin: process.env.CLIENT_URL || "*" }));
app.use(express.json());

// ── Routes ────────────────────────────────────────────────────────────────────

// GET /api/signals — current z-scores for all pairs
app.get("/api/signals", async (req, res) => {
  try {
    const docs = await db.collection("signals")
      .find({}, { projection: { _id: 0 } })
      .toArray();
    res.json(docs);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// GET /api/pairs — tradeable pairs + cointegration stats
app.get("/api/pairs", async (req, res) => {
  try {
    const docs = await db.collection("cointegration")
      .find({ tradeable: true }, { projection: { _id: 0 } })
      .toArray();
    res.json(docs);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// GET /api/trades — full trade log
app.get("/api/trades", async (req, res) => {
  try {
    const docs = await db.collection("trades")
      .find({}, { projection: { _id: 0 } })
      .sort({ exit_date: -1 })
      .toArray();
    // Serialize dates as ISO strings
    const serialized = docs.map(d => ({
      ...d,
      entry_date: d.entry_date?.toISOString?.() ?? d.entry_date,
      exit_date:  d.exit_date?.toISOString?.()  ?? d.exit_date,
    }));
    res.json(serialized);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// GET /api/backtest — portfolio performance metrics
app.get("/api/backtest", async (req, res) => {
  try {
    const doc = await db.collection("backtest_reports")
      .findOne({ report_type: "full_portfolio" },
               { projection: { _id: 0 } });
    res.json(doc || {});
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// GET /api/hmm — HMM regime filter comparison
app.get("/api/hmm", async (req, res) => {
  try {
    const doc = await db.collection("hmm_results")
      .findOne({ report_type: "hmm_comparison" },
               { projection: { _id: 0 } });
    res.json(doc || {});
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// GET /api/kmeans — ML-discovered pairs
app.get("/api/kmeans", async (req, res) => {
  try {
    const docs = await db.collection("kmeans_pairs")
      .find({ tradeable: true }, { projection: { _id: 0 } })
      .toArray();
    res.json(docs);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Health check
app.get("/api/health", (_, res) =>
  res.json({ status: "ok", ts: new Date().toISOString() }));

// ── Start ─────────────────────────────────────────────────────────────────────
connectDB().then(() => {
  app.listen(PORT, () =>
    console.log(`API running on http://localhost:${PORT}`));
}).catch(err => {
  console.error("Failed to connect to MongoDB:", err);
  process.exit(1);
})