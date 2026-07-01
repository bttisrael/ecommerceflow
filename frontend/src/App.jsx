import React, { useEffect, useState } from 'react';
import {
  Package,
  DollarSign,
  Truck,
  AlertTriangle,
  Play,
  BrainCircuit,
  Activity,
} from 'lucide-react';
import './style.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const DEFAULT_BATCH_SIZE = 3000;
const DEFAULT_INTERVAL_MINUTES = 240;

function MetricCard({ icon, label, value }) {
  return (
    <div className="metric-card">
      <div className="metric-icon">{icon}</div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function App() {
  const [orders, setOrders] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [status, setStatus] = useState(null);

  const [predictionSummary, setPredictionSummary] = useState(null);
  const [predictionPerformance, setPredictionPerformance] = useState(null);
  const [latestPredictions, setLatestPredictions] = useState([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function fetchJson(url, options = {}) {
    const res = await fetch(url, options);

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`${res.status} ${res.statusText}: ${text}`);
    }

    return res.json();
  }

  async function fetchData() {
    try {
      setError('');

      const [
        ordersData,
        metricsData,
        statusData,
        predictionSummaryData,
        predictionPerformanceData,
        latestPredictionsData,
      ] = await Promise.all([
        fetchJson(`${API_URL}/orders?limit=100&offset=0`),
        fetchJson(`${API_URL}/metrics`),
        fetchJson(`${API_URL}/simulation/status`),
        fetchJson(`${API_URL}/predictions/summary`),
        fetchJson(`${API_URL}/predictions/performance?limit=14`),
        fetchJson(`${API_URL}/predictions/latest?limit=100`),
      ]);

      setOrders(ordersData.orders || []);
      setMetrics(metricsData);
      setStatus(statusData);
      setPredictionSummary(predictionSummaryData);
      setPredictionPerformance(predictionPerformanceData);
      setLatestPredictions(latestPredictionsData.predictions || []);
    } catch (err) {
      console.error(err);
      setError(`API error: ${err.message}`);
    }
  }

  async function generateBatch() {
    try {
      setLoading(true);
      setError('');

      const batchSize = status?.orders_per_batch || DEFAULT_BATCH_SIZE;
      await fetchJson(`${API_URL}/orders/generate?n=${batchSize}`, {
        method: 'POST',
      });

      await fetchData();
    } catch (err) {
      console.error(err);
      setError(`Generate failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchData();

    const timer = setInterval(fetchData, 30000);

    return () => clearInterval(timer);
  }, []);

  const fmtMoney = (v) =>
    Number(v || 0).toLocaleString('pt-BR', {
      style: 'currency',
      currency: 'BRL',
    });

  const fmtNum = (v) =>
    Number(v || 0).toLocaleString('pt-BR', {
      maximumFractionDigits: 1,
    });

  const fmtPct = (v) => `${(Number(v || 0) * 100).toFixed(1)}%`;

  const fmtPctMaybe = (v) => (v === null || v === undefined ? 'not available' : fmtPct(v));

  const fmtDelta = (v) => {
    if (v === null || v === undefined) return 'no baseline';
    const sign = Number(v) > 0 ? '+' : '';
    return `${sign}${(Number(v) * 100).toFixed(1)} pp`;
  };

  const deltaClass = (v) => {
    if (v === null || v === undefined) return 'metric-delta neutral';
    if (Number(v) < 0) return 'metric-delta down';
    if (Number(v) > 0) return 'metric-delta up';
    return 'metric-delta neutral';
  };

  const riskClass = (riskBand) => {
    if (riskBand === 'high') return 'risk high';
    if (riskBand === 'medium') return 'risk medium';
    return 'risk low';
  };

  const latestPerformance = predictionPerformance?.latest;
  const performanceHistory = predictionPerformance?.history || [];
  const performanceAlert = latestPerformance?.accuracy_drop_alert || latestPerformance?.f1_drop_alert;

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">real-time ecommerce simulator</p>
          <h1>CommerceFlow AI</h1>
          <p className="subtitle">
            Synthetic e-commerce orders generated every 4 hours, ingested through an API,
            processed in BigQuery, and scored by a Vertex AI delay-risk model.
          </p>
        </div>

        <button className="generate-btn" onClick={generateBatch} disabled={loading}>
          <Play size={18} />{' '}
          {loading ? 'Generating...' : `Generate ${status?.orders_per_batch || DEFAULT_BATCH_SIZE} Orders`}
        </button>
      </header>

      {error && (
        <div
          style={{
            color: '#fca5a5',
            border: '1px solid #ef4444',
            padding: '12px',
            borderRadius: '8px',
            marginBottom: '16px',
          }}
        >
          {error}
        </div>
      )}

      <section className="status-bar">
        <span className={status?.job_exists ? 'dot on' : 'dot'}></span>
        Scheduler: {status?.job_exists ? 'active' : 'inactive'} · batch size:{' '}
        {status?.orders_per_batch || DEFAULT_BATCH_SIZE} · interval:{' '}
        {status?.interval_minutes || DEFAULT_INTERVAL_MINUTES} min · next run:{' '}
        {status?.next_run_time || 'not scheduled'}
      </section>

      <section className="metrics-grid">
        <MetricCard
          icon={<Package />}
          label="Total Orders"
          value={fmtNum(metrics?.total_orders)}
        />

        <MetricCard
          icon={<DollarSign />}
          label="Total Revenue"
          value={fmtMoney(metrics?.total_revenue)}
        />

        <MetricCard
          icon={<Truck />}
          label="Avg Distance"
          value={`${fmtNum(metrics?.avg_distance_km)} km`}
        />

        <MetricCard
          icon={<AlertTriangle />}
          label="Synthetic Delay Rate"
          value={fmtPct(metrics?.delay_risk_rate)}
        />

        <MetricCard
          icon={<BrainCircuit />}
          label="Vertex Avg Risk"
          value={fmtPct(predictionSummary?.avg_delay_probability)}
        />

        <MetricCard
          icon={<Activity />}
          label="Vertex High Risk"
          value={Number(predictionSummary?.high_risk_orders || 0).toLocaleString('pt-BR')}
        />

        <MetricCard
          icon={<Activity />}
          label="Latest Accuracy"
          value={fmtPctMaybe(latestPerformance?.accuracy)}
        />

        <MetricCard
          icon={<AlertTriangle />}
          label="Accuracy Change"
          value={fmtDelta(latestPerformance?.accuracy_delta)}
        />
      </section>

      <section className="grid-two">
        <div className="panel">
          <h2>Orders by State</h2>

          {(metrics?.orders_by_state || []).slice(0, 8).map((r) => (
            <div className="bar-row" key={r.state}>
              <span>{r.state}</span>

              <div className="bar">
                <div
                  style={{
                    width: `${Math.min(
                      100,
                      (r.orders / Math.max(1, metrics?.total_orders || 1)) * 300
                    )}%`,
                  }}
                ></div>
              </div>

              <strong>{r.orders}</strong>
            </div>
          ))}
        </div>

        <div className="panel">
          <h2>Carrier Risk</h2>

          {(metrics?.orders_by_carrier || []).map((r) => (
            <div className="carrier-row" key={r.carrier}>
              <span>{r.carrier}</span>
              <strong>{fmtPct(r.delay_risk_rate)}</strong>
            </div>
          ))}
        </div>
      </section>

      <section className="grid-two">
        <div className="panel">
          <h2>Vertex Risk Distribution</h2>

          {(predictionSummary?.risk_distribution || []).map((r) => (
            <div className="carrier-row" key={r.risk_band}>
              <span>
                <span className={riskClass(r.risk_band)}>{r.risk_band}</span>
              </span>

              <strong>
                {Number(r.orders || 0).toLocaleString('pt-BR')} orders ·{' '}
                {fmtPct(r.avg_delay_probability)}
              </strong>
            </div>
          ))}
        </div>

        <div className="panel">
          <h2>Vertex Scoring Status</h2>

          <div className="carrier-row">
            <span>Total scored orders</span>
            <strong>
              {Number(predictionSummary?.total_predictions || 0).toLocaleString('pt-BR')}
            </strong>
          </div>

          <div className="carrier-row">
            <span>High risk orders</span>
            <strong>
              {Number(predictionSummary?.high_risk_orders || 0).toLocaleString('pt-BR')}
            </strong>
          </div>

          <div className="carrier-row">
            <span>Medium risk orders</span>
            <strong>
              {Number(predictionSummary?.medium_risk_orders || 0).toLocaleString('pt-BR')}
            </strong>
          </div>

          <div className="carrier-row">
            <span>Low risk orders</span>
            <strong>
              {Number(predictionSummary?.low_risk_orders || 0).toLocaleString('pt-BR')}
            </strong>
          </div>

          <div className="carrier-row">
            <span>Last prediction</span>
            <strong>
              {predictionSummary?.last_prediction_timestamp
                ? new Date(predictionSummary.last_prediction_timestamp).toLocaleString()
                : 'not available'}
            </strong>
          </div>
        </div>
      </section>

      <section className="grid-two">
        <div className={performanceAlert ? 'panel panel-alert' : 'panel'}>
          <h2>Prediction Performance</h2>

          <div className="carrier-row">
            <span>Accuracy</span>
            <strong>
              {fmtPctMaybe(latestPerformance?.accuracy)}{' '}
              <span className={deltaClass(latestPerformance?.accuracy_delta)}>
                {fmtDelta(latestPerformance?.accuracy_delta)}
              </span>
            </strong>
          </div>

          <div className="carrier-row">
            <span>F1</span>
            <strong>
              {fmtPctMaybe(latestPerformance?.f1)}{' '}
              <span className={deltaClass(latestPerformance?.f1_delta)}>
                {fmtDelta(latestPerformance?.f1_delta)}
              </span>
            </strong>
          </div>

          <div className="carrier-row">
            <span>Precision</span>
            <strong>
              {fmtPctMaybe(latestPerformance?.precision)}{' '}
              <span className={deltaClass(latestPerformance?.precision_delta)}>
                {fmtDelta(latestPerformance?.precision_delta)}
              </span>
            </strong>
          </div>

          <div className="carrier-row">
            <span>Recall</span>
            <strong>
              {fmtPctMaybe(latestPerformance?.recall)}{' '}
              <span className={deltaClass(latestPerformance?.recall_delta)}>
                {fmtDelta(latestPerformance?.recall_delta)}
              </span>
            </strong>
          </div>

          <div className="carrier-row">
            <span>Evaluated rows</span>
            <strong>
              {Number(latestPerformance?.evaluated_rows || 0).toLocaleString('pt-BR')}
            </strong>
          </div>

          <div className="carrier-row">
            <span>Baseline</span>
            <strong>
              {latestPerformance?.baseline_run_timestamp
                ? new Date(latestPerformance.baseline_run_timestamp).toLocaleString()
                : 'not available'}
            </strong>
          </div>
        </div>

        <div className="panel">
          <h2>Performance History</h2>

          {performanceHistory.slice(0, 6).map((run) => (
            <div className="history-row" key={run.run_id}>
              <span>
                {run.run_timestamp ? new Date(run.run_timestamp).toLocaleDateString() : 'not available'}
              </span>

              <strong>{fmtPct(run.accuracy)}</strong>

              <span className={deltaClass(run.accuracy_delta)}>
                {fmtDelta(run.accuracy_delta)}
              </span>
            </div>
          ))}

          {performanceHistory.length === 0 && (
            <div className="empty-state">No performance runs yet</div>
          )}
        </div>
      </section>

      <section className="panel table-panel">
        <div className="table-header">
          <h2>Latest Orders</h2>
          <span>Auto-refreshes every 30 seconds</span>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Customer</th>
                <th>City</th>
                <th>Product</th>
                <th>Value</th>
                <th>Distance</th>
                <th>Carrier</th>
                <th>Synthetic Risk</th>
              </tr>
            </thead>

            <tbody>
              {orders.map((o) => (
                <tr key={o.order_id}>
                  <td>{new Date(o.order_timestamp).toLocaleString()}</td>

                  <td>
                    <b>{o.customer_name}</b>
                    <br />
                    <small>{o.customer_email}</small>
                  </td>

                  <td>
                    {o.city}/{o.state}
                  </td>

                  <td>
                    {o.product_bought}
                    <br />
                    <small>{o.product_category}</small>
                  </td>

                  <td>{fmtMoney(o.order_value)}</td>
                  <td>{fmtNum(o.distance_km)} km</td>
                  <td>{o.carrier}</td>

                  <td>
                    <span className={o.delay_risk_label ? 'risk high' : 'risk low'}>
                      {o.delay_risk_label ? 'high' : 'low'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel table-panel">
        <div className="table-header">
          <h2>Latest Vertex AI Predictions</h2>
          <span>Scored by Vertex AI Endpoint</span>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Prediction Time</th>
                <th>Order ID</th>
                <th>Delay Probability</th>
                <th>Prediction</th>
                <th>Risk Band</th>
                <th>Model</th>
              </tr>
            </thead>

            <tbody>
              {latestPredictions.map((p) => (
                <tr key={`${p.order_id}-${p.prediction_timestamp}`}>
                  <td>{new Date(p.prediction_timestamp).toLocaleString()}</td>
                  <td>{p.order_id}</td>
                  <td>{fmtPct(p.delay_probability)}</td>
                  <td>{p.delay_prediction ? 'Delay' : 'On time'}</td>

                  <td>
                    <span className={riskClass(p.risk_band)}>{p.risk_band}</span>
                  </td>

                  <td>{p.model_version}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

export default App;
