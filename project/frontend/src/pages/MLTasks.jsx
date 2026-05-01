import { useState, useEffect } from 'react';
import { TrendingUp, Clock, Layers, GitBranch, Link, BarChart2, RefreshCw } from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter, Cell
} from 'recharts';

const API_BASE = import.meta.env.DEV ? 'http://localhost:8000' : '';

const EmptyState = ({ msg }) => (
  <div className="flex-col items-center justify-center" style={{ height: 200, opacity: 0.5 }}>
    <BarChart2 size={32} style={{ marginBottom: '0.5rem' }} />
    <span className="text-sm text-muted">{msg}</span>
  </div>
);

const SectionHeader = ({ icon: Icon, color, label, title }) => (
  <div className="flex-col gap-1" style={{ marginBottom: '0.5rem' }}>
    <span className="label" style={{ color }}>{label}</span>
    <h2 className="flex items-center gap-2" style={{ margin: 0 }}>
      <Icon size={20} style={{ color }} /> {title}
    </h2>
  </div>
);

const MetricPill = ({ label, value, color }) => (
  <div className="glass-card flex-col gap-1" style={{ padding: '1rem 1.25rem', minWidth: 120 }}>
    <span className="label" style={{ color: 'var(--text-muted)' }}>{label}</span>
    <span style={{ fontSize: '1.6rem', fontWeight: 800, color: color || 'var(--text-primary)', letterSpacing: '-0.03em' }}>
      {value}
    </span>
  </div>
);

// ── 1. Regression ─────────────────────────────────────────────────────────────
const RegressionSection = ({ data }) => {
  if (!data || data.available === false)
    return <EmptyState msg="Run the pipeline to generate regression results." />;

  const forecast = (data.forecast || []).slice(0, 100);
  return (
    <div className="flex-col gap-5">
      <div className="flex-row gap-4" style={{ flexWrap: 'wrap' }}>
        <MetricPill label="RMSE" value={data.rmse?.toFixed(4)} color="var(--cyan)" />
        <MetricPill label="MAE"  value={data.mae?.toFixed(4)}  color="var(--amber)" />
        <MetricPill label="R²"   value={data.r2?.toFixed(4)}   color="var(--emerald)" />
        <MetricPill label="Train Windows" value={data.train_windows} />
        <MetricPill label="Test Windows"  value={data.test_windows} />
      </div>
      {forecast.length > 0 && (
        <div className="glass-card p-5" style={{ height: 280 }}>
          <h3 className="flex items-center gap-2 mb-4"><TrendingUp size={15} className="text-cyan" /> Actual vs Forecast (Test Set)</h3>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={forecast}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
              <XAxis dataKey="window" stroke="var(--text-muted)" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-default)', borderRadius: 8 }} />
              <Line type="monotone" dataKey="actual"    stroke="var(--cyan)"  strokeWidth={2} dot={false} name="Actual" />
              <Line type="monotone" dataKey="predicted" stroke="var(--rose)"  strokeWidth={2} dot={false} strokeDasharray="5 3" name="Predicted" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      <div className="glass-card p-4">
        <p className="text-secondary text-sm" style={{ margin: 0 }}>
          <strong style={{ color: 'var(--text-primary)' }}>Model:</strong> {data.model} &nbsp;|&nbsp;
          <strong style={{ color: 'var(--text-primary)' }}>Window:</strong> {data.window_hours}h &nbsp;|&nbsp;
          <strong style={{ color: 'var(--text-primary)' }}>Lag features:</strong> {data.n_lags}
        </p>
      </div>
    </div>
  );
};

// ── 2. Time Series ────────────────────────────────────────────────────────────
const TimeSeriesSection = ({ data }) => {
  if (!data || data.available === false)
    return <EmptyState msg="Run the pipeline to generate time series results." />;

  const ts = (data.timeseries || []).slice(0, 300);
  const anomalies = ts.filter(d => d.is_anomaly);
  return (
    <div className="flex-col gap-5">
      <div className="flex-row gap-4" style={{ flexWrap: 'wrap' }}>
        <MetricPill label="Overall Fraud Rate"  value={(data.overall_fraud_rate * 100).toFixed(3) + '%'} color="var(--rose)" />
        <MetricPill label="Peak Fraud Rate"     value={(data.peak_fraud_rate * 100).toFixed(3) + '%'}   color="var(--amber)" />
        <MetricPill label="Anomalous Windows"   value={data.anomalous_windows} color="var(--rose)" />
        <MetricPill label="Total Windows"       value={data.total_windows} />
      </div>
      {ts.length > 0 && (
        <div className="glass-card p-5" style={{ height: 280 }}>
          <h3 className="flex items-center gap-2 mb-4"><Clock size={15} className="text-emerald" /> Fraud Rate Over Time</h3>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={ts}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
              <XAxis dataKey="window" stroke="var(--text-muted)" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-default)', borderRadius: 8 }} />
              <Line type="monotone" dataKey="fraud_rate"  stroke="var(--cyan)"    strokeWidth={1.5} dot={false} name="Fraud Rate" />
              <Line type="monotone" dataKey="rolling_avg" stroke="var(--emerald)" strokeWidth={2.5} dot={false} name="Rolling Avg" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      {anomalies.length > 0 && (
        <div className="glass-card p-4">
          <h3 className="flex items-center gap-2 mb-3"><span className="badge badge-rose">⚠ {anomalies.length} Anomalies</span></h3>
          <div className="flex-row gap-2" style={{ flexWrap: 'wrap' }}>
            {anomalies.slice(0, 10).map((a, i) => (
              <span key={i} className="badge badge-rose">Window {a.window} — {(a.fraud_rate * 100).toFixed(2)}%</span>
            ))}
            {anomalies.length > 10 && <span className="text-muted text-sm">+{anomalies.length - 10} more</span>}
          </div>
        </div>
      )}
    </div>
  );
};

// ── 3. PCA ────────────────────────────────────────────────────────────────────
const PCASection = ({ data }) => {
  if (!data || data.available === false)
    return <EmptyState msg="Run the pipeline to generate PCA results." />;

  const scatter = data.scatter_sample || [];
  const fraud  = scatter.filter(d => d.is_fraud === 1);
  const safe   = scatter.filter(d => d.is_fraud === 0);
  const varExp = (data.variance_explained || []).map((v, i) => ({ pc: `PC${i + 1}`, variance: +(v * 100).toFixed(2) }));

  return (
    <div className="flex-col gap-5">
      <div className="flex-row gap-4" style={{ flexWrap: 'wrap' }}>
        <MetricPill label="Total Features"   value={data.total_features}  color="var(--violet)" />
        <MetricPill label="Components"       value={data.n_components} />
        <MetricPill label="PC1 Variance"     value={(data.variance_explained?.[0] * 100).toFixed(1) + '%'} color="var(--cyan)" />
        <MetricPill label="PC2 Variance"     value={(data.variance_explained?.[1] * 100).toFixed(1) + '%'} color="var(--amber)" />
        <MetricPill label="Cumulative Top-N" value={(data.cumulative_variance?.slice(-1)[0] * 100).toFixed(1) + '%'} color="var(--emerald)" />
      </div>
      <div className="grid-2 gap-5">
        {scatter.length > 0 && (
          <div className="glass-card p-5" style={{ height: 300 }}>
            <h3 className="flex items-center gap-2 mb-3"><Layers size={15} className="text-violet" /> 2D Scatter (Fraud vs Safe)</h3>
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
                <XAxis dataKey="pc1" name="PC1" stroke="var(--text-muted)" tick={{ fontSize: 10 }} axisLine={false} tickLine={false} />
                <YAxis dataKey="pc2" name="PC2" stroke="var(--text-muted)" tick={{ fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-default)', borderRadius: 8 }} />
                <Scatter name="Safe"  data={safe.slice(0, 800)}  fill="var(--cyan)"  fillOpacity={0.3} />
                <Scatter name="Fraud" data={fraud.slice(0, 400)} fill="var(--rose)"  fillOpacity={0.7} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        )}
        {varExp.length > 0 && (
          <div className="glass-card p-5" style={{ height: 300 }}>
            <h3 className="flex items-center gap-2 mb-3"><BarChart2 size={15} className="text-amber" /> Variance Explained per PC</h3>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={varExp}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
                <XAxis dataKey="pc" stroke="var(--text-muted)" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} unit="%" />
                <Tooltip contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-default)', borderRadius: 8 }} />
                <Bar dataKey="variance" fill="var(--amber)" radius={[4, 4, 0, 0]} name="Variance %" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
};

// ── 4. Clustering ─────────────────────────────────────────────────────────────
const ClusteringSection = ({ data }) => {
  if (!data || data.available === false)
    return <EmptyState msg="Run the pipeline to generate cluster profiles." />;

  const profiles = data.profiles || [];
  const overallRate = data.overall_fraud_rate * 100;
  const chartData = profiles.map(p => ({ name: `Cluster ${p.cluster}`, fraud_rate: +(p.fraud_rate * 100).toFixed(3), size: p.size }));

  return (
    <div className="flex-col gap-5">
      <div className="flex-row gap-4" style={{ flexWrap: 'wrap' }}>
        <MetricPill label="Clusters"           value={data.n_clusters} color="var(--violet)" />
        <MetricPill label="Overall Fraud Rate" value={overallRate.toFixed(3) + '%'} color="var(--rose)" />
      </div>
      {profiles.length > 0 && (
        <div className="glass-card p-5" style={{ height: 280 }}>
          <h3 className="flex items-center gap-2 mb-4"><GitBranch size={15} className="text-violet" /> Fraud Rate by Cluster</h3>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
              <XAxis dataKey="name" stroke="var(--text-muted)" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} unit="%" />
              <Tooltip contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-default)', borderRadius: 8 }} />
              <Bar dataKey="fraud_rate" radius={[4, 4, 0, 0]} name="Fraud Rate %">
                {chartData.map((entry, i) => (
                  <Cell key={i} fill={entry.fraud_rate > overallRate ? 'var(--rose)' : 'var(--violet)'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead><tr><th>Cluster</th><th>Size</th><th>% of Data</th><th>Fraud Count</th><th>Fraud Rate</th><th>vs Overall</th></tr></thead>
          <tbody>
            {profiles.map((p, i) => (
              <tr key={i}>
                <td className="mono text-violet font-medium">Cluster {p.cluster}</td>
                <td>{p.size.toLocaleString()}</td>
                <td>{p.pct_of_data}%</td>
                <td>{p.fraud_count.toLocaleString()}</td>
                <td style={{ color: p.fraud_rate * 100 > overallRate ? 'var(--rose)' : 'var(--emerald)', fontWeight: 600 }}>
                  {(p.fraud_rate * 100).toFixed(3)}%
                </td>
                <td>
                  <span className={`badge ${p.fraud_rate * 100 > overallRate ? 'badge-rose' : 'badge-emerald'}`}>
                    {p.fraud_rate * 100 > overallRate ? '↑ High Risk' : '↓ Low Risk'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data.insight && (
        <div className="glass-card p-4" style={{ borderLeft: '3px solid var(--amber)' }}>
          <span className="text-secondary text-sm">💡 {data.insight}</span>
        </div>
      )}
    </div>
  );
};

// ── 5. Association Rules ──────────────────────────────────────────────────────
const AssociationSection = ({ data }) => {
  if (!data || data.available === false)
    return <EmptyState msg="Run the pipeline to generate association rules." />;

  const rules = data.top_rules || [];
  return (
    <div className="flex-col gap-5">
      <div className="flex-row gap-4" style={{ flexWrap: 'wrap' }}>
        <MetricPill label="Frequent Itemsets" value={data.n_itemsets}     color="var(--cyan)" />
        <MetricPill label="Fraud Rules Found" value={data.n_fraud_rules}  color="var(--rose)" />
        <MetricPill label="Min Support"       value={data.min_support} />
        <MetricPill label="Min Confidence"    value={data.min_confidence} />
      </div>
      {rules.length > 0 ? (
        <div style={{ overflowX: 'auto' }}>
          <table>
            <thead>
              <tr><th>#</th><th>Antecedents (If...)</th><th>Consequent</th><th>Support</th><th>Confidence</th><th>Lift</th></tr>
            </thead>
            <tbody>
              {rules.map((r, i) => (
                <tr key={i}>
                  <td className="text-muted text-sm">{i + 1}</td>
                  <td>
                    <div className="flex-row gap-1" style={{ flexWrap: 'wrap' }}>
                      {r.antecedents.map((a, j) => (
                        <span key={j} className="badge badge-cyan">{a}</span>
                      ))}
                    </div>
                  </td>
                  <td>
                    <span className="badge badge-rose">{r.consequents[0]}</span>
                  </td>
                  <td className="mono text-sm">{(r.support * 100).toFixed(2)}%</td>
                  <td className="mono text-sm">{(r.confidence * 100).toFixed(1)}%</td>
                  <td>
                    <span style={{ fontWeight: 700, color: r.lift > 2 ? 'var(--rose)' : r.lift > 1.5 ? 'var(--amber)' : 'var(--emerald)' }}>
                      {r.lift.toFixed(3)}×
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <EmptyState msg="No fraud-consequent rules found with current thresholds." />
      )}
    </div>
  );
};

// ── Main Page ─────────────────────────────────────────────────────────────────
const TABS = [
  { key: 'regression',   label: 'Regression',    icon: TrendingUp, color: 'var(--cyan)',    endpoint: '/regression_results' },
  { key: 'timeseries',   label: 'Time Series',   icon: Clock,      color: 'var(--emerald)', endpoint: '/timeseries_results' },
  { key: 'pca',          label: 'Dim. Reduction', icon: Layers,     color: 'var(--amber)',   endpoint: '/pca_results' },
  { key: 'clustering',   label: 'Clustering',    icon: GitBranch,  color: 'var(--violet)',  endpoint: '/cluster_profiles' },
  { key: 'association',  label: 'Association',   icon: Link,       color: 'var(--rose)',    endpoint: '/association_rules' },
];

const MLTasks = () => {
  const [activeTab, setActiveTab]   = useState('regression');
  const [taskData, setTaskData]     = useState({});
  const [loading, setLoading]       = useState(false);

  useEffect(() => {
    const fetchAll = async () => {
      setLoading(true);
      const results = await Promise.allSettled(
        TABS.map(t => fetch(`${API_BASE}${t.endpoint}`).then(r => r.json()))
      );
      const newData = {};
      TABS.forEach((t, i) => {
        if (results[i].status === 'fulfilled') newData[t.key] = results[i].value;
      });
      setTaskData(newData);
      setLoading(false);
    };
    fetchAll();
  }, []);

  const activeTabMeta = TABS.find(t => t.key === activeTab);

  return (
    <div className="scroll-container flex-col gap-8">
      {/* Header */}
      <header className="flex-row justify-between items-start animate-in delay-1">
        <div className="flex-col gap-1">
          <span className="label text-cyan">Multi-Task Intelligence</span>
          <h1>ML Analytics Suite</h1>
          <p className="text-secondary mt-2" style={{ maxWidth: 580 }}>
            Five independent analytical branches — regression, time series, dimensionality
            reduction, clustering, and association rules — all derived from the same fraud dataset.
          </p>
        </div>
        <button
          id="refresh-mltasks-btn"
          onClick={() => window.location.reload()}
          disabled={loading}
          style={{
            display: 'flex', alignItems: 'center', gap: '0.5rem',
            padding: '0.5rem 1.2rem', borderRadius: '8px',
            border: '1px solid var(--border-default)',
            background: 'var(--bg-elevated)',
            color: loading ? 'var(--text-muted)' : 'var(--text-primary)',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '0.875rem', fontWeight: 500,
          }}
        >
          <RefreshCw size={15} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
          {loading ? 'Loading...' : 'Refresh'}
        </button>
      </header>

      {/* Tab Strip */}
      <div className="flex-row gap-2 animate-in delay-2" style={{ flexWrap: 'wrap' }}>
        {TABS.map(tab => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.key;
          return (
            <button
              key={tab.key}
              id={`mltask-tab-${tab.key}`}
              onClick={() => setActiveTab(tab.key)}
              style={{
                display: 'flex', alignItems: 'center', gap: '0.5rem',
                padding: '0.55rem 1.1rem', borderRadius: '8px',
                border: `1px solid ${isActive ? tab.color : 'var(--border-default)'}`,
                background: isActive ? `color-mix(in srgb, ${tab.color} 10%, transparent)` : 'var(--bg-elevated)',
                color: isActive ? tab.color : 'var(--text-secondary)',
                cursor: 'pointer', fontSize: '0.875rem', fontWeight: isActive ? 600 : 400,
                transition: 'all 0.2s ease',
              }}
            >
              <Icon size={15} /> {tab.label}
            </button>
          );
        })}
      </div>

      {/* Active Panel */}
      <div className="glass-card p-6 flex-col gap-6 animate-in delay-3">
        <SectionHeader
          icon={activeTabMeta.icon}
          color={activeTabMeta.color}
          label={`Task — ${activeTabMeta.label}`}
          title={{
            regression:  'Transaction Velocity Forecasting (Ridge Regression)',
            timeseries:  'Fraud Rate Time Series Analysis',
            pca:         'Dimensionality Reduction (PCA)',
            clustering:  'Behavioural Cluster Profiling (KMeans)',
            association: 'Fraud Pattern Association Rules (FPGrowth)',
          }[activeTab]}
        />

        {loading ? (
          <div className="flex-col items-center justify-center" style={{ height: 200 }}>
            <RefreshCw size={28} style={{ animation: 'spin 1s linear infinite', color: 'var(--text-muted)' }} />
            <span className="text-muted text-sm mt-2">Fetching results...</span>
          </div>
        ) : (
          <>
            {activeTab === 'regression'  && <RegressionSection  data={taskData.regression} />}
            {activeTab === 'timeseries'  && <TimeSeriesSection  data={taskData.timeseries} />}
            {activeTab === 'pca'         && <PCASection         data={taskData.pca} />}
            {activeTab === 'clustering'  && <ClusteringSection  data={taskData.clustering} />}
            {activeTab === 'association' && <AssociationSection data={taskData.association} />}
          </>
        )}
      </div>

      {/* PNG Gallery for ML Task graphs */}
      <div className="glass-card p-6 flex-col gap-4 animate-in delay-4">
        <h3 className="flex items-center gap-2"><BarChart2 size={16} className="text-amber" /> Result Plots Gallery</h3>
        <p className="text-secondary text-sm" style={{ margin: 0 }}>
          Static PNG outputs saved by the pipeline for all ML task modules.
        </p>
        <div className="grid-2" style={{ gap: '1rem' }}>
          {[
            { file: 'latest_regression_forecast.png', label: '📈 Regression Forecast' },
            { file: 'latest_timeseries_line.png',     label: '🕐 Time Series Line Chart' },
            { file: 'latest_timeseries_heatmap.png',  label: '🌡 Fraud Heatmap' },
            { file: 'latest_pca_scatter.png',         label: '🔵 PCA 2D Scatter' },
            { file: 'latest_pca_variance.png',        label: '📊 PCA Variance' },
            { file: 'latest_cluster_fraud_rate.png',  label: '🏷 Cluster Fraud Rates' },
            { file: 'latest_cluster_size.png',        label: '🥧 Cluster Sizes' },
            { file: 'latest_association_lift.png',    label: '🔗 Association Lift' },
          ].map(({ file, label }) => (
            <div key={file} className="glass-card flex-col" style={{ overflow: 'hidden' }}>
              <div style={{ padding: '0.6rem 1rem', borderBottom: '1px solid var(--border-subtle)' }}>
                <span className="label">{label}</span>
              </div>
              <img
                src={`${API_BASE}/graphs/${file}`}
                alt={label}
                style={{ width: '100%', objectFit: 'contain', maxHeight: 260, background: 'var(--bg-elevated)' }}
                onError={e => { e.target.style.display = 'none'; }}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default MLTasks;
