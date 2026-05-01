import { useState, useEffect, useCallback } from 'react';
import { Activity, Database, GitCommit, Play, CheckCircle2, Image as ImageIcon, BarChart2, RefreshCw, XCircle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const API_BASE = import.meta.env.DEV ? 'http://localhost:8000' : '';

// Human-readable labels for each graph filename returned by /graph_list
const GRAPH_LABELS = {
  'roc_curve.png':         '📈 ROC Curve',
  'pr_curve.png':          '📉 Precision-Recall Curve',
  'confusion_matrix.png':  '🟦 Confusion Matrix',
  'metric_summary.png':    '📊 Metric Summary',
};

const Dashboard = () => {
  const [stats, setStats] = useState({ total: 0, fraud: 0, safe: 0, avgProb: 0 });
  const [evaluations, setEvaluations] = useState([]);
  const [curveData, setCurveData] = useState({ roc_curve: [], pr_curve: [], cache_bust: '' });
  const [graphs, setGraphs] = useState([]);
  const [activeGraph, setActiveGraph] = useState('');
  const [loading, setLoading] = useState(false);
  const [lastRefresh, setLastRefresh] = useState(null);
  const [pipelineStatus, setPipelineStatus] = useState({ status: 'IDLE' });

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      // Stats from localStorage
      const history = JSON.parse(localStorage.getItem('predictionHistory') || '[]');
      if (history.length > 0) {
        const total = history.length;
        const fraud = history.filter(item => item.is_fraud === 1).length;
        const avgProb = history.reduce((acc, curr) => acc + curr.probability, 0) / total;
        setStats({ total, fraud, safe: total - fraud, avgProb: (avgProb * 100).toFixed(1) });
      }

      const [evalRes, metricsRes, graphRes, statusRes] = await Promise.allSettled([
        fetch(`${API_BASE}/model_evaluations`).then(r => r.json()),
        fetch(`${API_BASE}/latest_metrics`).then(r => r.json()),
        fetch(`${API_BASE}/graph_list`).then(r => r.json()),
        fetch(`${API_BASE}/pipeline/status`).then(r => r.json()),
      ]);

      if (evalRes.status === 'fulfilled' && evalRes.value.evaluations)
        setEvaluations(evalRes.value.evaluations.reverse());

      if (metricsRes.status === 'fulfilled' && metricsRes.value.roc_curve)
        setCurveData(metricsRes.value);

      if (graphRes.status === 'fulfilled' && graphRes.value.graphs?.length > 0) {
        setGraphs(graphRes.value.graphs);
        setActiveGraph(g => g || graphRes.value.graphs[0]);
      }

      if (statusRes.status === 'fulfilled') {
        setPipelineStatus(statusRes.value);
      }

      setLastRefresh(new Date());
    } catch (err) {
      console.error('Fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleCancel = async (runId) => {
    if (!window.confirm("Are you sure you want to terminate this training run?")) return;
    try {
      const res = await fetch(`${API_BASE}/pipeline/cancel/${runId}`, { method: 'POST' });
      if (res.ok) {
        setPipelineStatus({ status: 'IDLE' });
        fetchData();
      }
    } catch (e) {
      console.error("Cancellation failed", e);
    }
  };

  useEffect(() => {
    fetchData();
    // Auto-refresh every 30s if running
    const interval = setInterval(() => {
      fetchData();
    }, 15000);
    return () => clearInterval(interval);
  }, [fetchData]);

  return (
    <div className="scroll-container flex-col gap-8">
      <header className="flex-row justify-between items-start animate-in delay-1">
        <div className="flex-col gap-1">
          <span className="label text-violet">System Overview</span>
          <h1 className="gradient-text-violet">ML Pipeline Dashboard</h1>
          <p className="text-secondary mt-2" style={{ maxWidth: '600px' }}>
            Real-time view of model training, evaluation metrics, and result graphs.
          </p>
        </div>
        <div className="flex-col items-end gap-1">
          <button
            id="refresh-dashboard-btn"
            onClick={fetchData}
            disabled={loading}
            style={{
              display: 'flex', alignItems: 'center', gap: '0.5rem',
              padding: '0.5rem 1.2rem', borderRadius: '8px',
              border: '1px solid var(--border-default)',
              background: 'var(--bg-elevated)',
              color: loading ? 'var(--text-muted)' : 'var(--text-primary)',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontSize: '0.875rem', fontWeight: '500',
              transition: 'all 0.2s ease',
            }}
          >
            <RefreshCw size={15} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }}/>
            {loading ? 'Refreshing...' : 'Refresh'}
          </button>
          {lastRefresh && (
            <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
              Updated {lastRefresh.toLocaleTimeString()}
            </span>
          )}
        </div>
      </header>

      {/* ── Pipeline Status Banner ── */}
      {pipelineStatus.status === 'RUNNING' && (
        <div className="glass-card flex-row items-center gap-5 p-5 animate-in" 
             style={{ 
               background: 'rgba(245, 158, 11, 0.04)', 
               borderColor: 'rgba(245, 158, 11, 0.2)', 
               borderLeft: '4px solid var(--amber)',
               marginBottom: '-1rem' 
             }}>
          <div className="flex-col items-center justify-center" style={{ 
            width: 40, height: 40, borderRadius: '50%', background: 'rgba(245, 158, 11, 0.1)' 
          }}>
            <RefreshCw size={20} className="text-amber" style={{ animation: 'spin 2s linear infinite' }} />
          </div>
          <div className="flex-col gap-1 flex-1">
            <div className="flex-row items-center gap-2">
              <span className="label text-amber">Active Pipeline Run</span>
              <span className="text-xs text-muted" style={{ fontWeight: 400 }}>• Started {new Date(pipelineStatus.start_time).toLocaleTimeString()}</span>
            </div>
            <h4 style={{ margin: 0, fontSize: '1rem', color: 'var(--text-primary)' }}>
              {pipelineStatus.state}: <span style={{ fontWeight: 400 }}>{pipelineStatus.name}</span>
            </h4>
          </div>
          <div className="flex-row gap-3">
             <button 
               className="btn btn-ghost btn-sm" 
               onClick={() => handleCancel(pipelineStatus.id)}
               style={{ borderColor: 'rgba(244, 63, 94, 0.2)', color: 'var(--rose)' }}
             >
               <XCircle size={14} /> Cancel Build
             </button>
             <div className="badge badge-amber" style={{ background: 'rgba(245, 158, 11, 0.1)', color: 'var(--amber)' }}>
               Live Training
             </div>
          </div>
        </div>
      )}

      {/* KPI Section */}
      <div className="grid-4 animate-in delay-2">
        {[
          { label: 'Total Predictions', val: stats.total.toString(),    icon: <Activity size={20}/>,  color: 'var(--cyan)' },
          { label: 'Fraud Detected',    val: stats.fraud.toString(),     icon: <Play size={20}/>,      color: 'var(--rose)' },
          { label: 'Safe Transactions', val: stats.safe.toString(),      icon: <Database size={20}/>,  color: 'var(--emerald)' },
          { label: 'Avg Fraud Prob',    val: `${stats.avgProb}%`,        icon: <GitCommit size={20}/>, color: 'var(--amber)' },
        ].map((kpi, i) => (
          <div key={i} className="glass-card p-6 flex-col gap-4">
            <div className="flex-row items-center gap-3">
              <div style={{ color: kpi.color, padding: '0.5rem', background: `color-mix(in srgb, ${kpi.color} 10%, transparent)`, borderRadius: '8px' }}>
                {kpi.icon}
              </div>
              <span className="label" style={{ color: 'var(--text-secondary)' }}>{kpi.label}</span>
            </div>
            <div className="stat-number">{kpi.val}</div>
          </div>
        ))}
      </div>

      {/* Interactive Recharts */}
      <div className="grid-2 animate-in delay-3">
        <div className="glass-card p-6 flex-col gap-6" style={{ height: '350px' }}>
          <h3 className="flex items-center gap-2"><Activity size={16} className="text-cyan"/> Precision-Recall Curve</h3>
          <div className="flex-1">
            {curveData.pr_curve.length === 0 ? (
              <div className="flex-col items-center justify-center h-full text-muted" style={{ height: '100%' }}>
                <BarChart2 size={32} style={{ marginBottom: '0.5rem', opacity: 0.4 }}/>
                <span className="text-sm">Run the pipeline to generate curve data</span>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={curveData.pr_curve}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
                  <XAxis dataKey="recall"    type="number" domain={[0, 1]} stroke="var(--text-muted)" tick={{fontSize: 12}} axisLine={false} tickLine={false} />
                  <YAxis dataKey="precision" type="number" domain={[0, 1]} stroke="var(--text-muted)" tick={{fontSize: 12}} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-default)', borderRadius: '8px' }} itemStyle={{ color: 'var(--text-primary)' }} />
                  <Line type="monotone" dataKey="precision" stroke="var(--cyan)" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        <div className="glass-card p-6 flex-col gap-6" style={{ height: '350px' }}>
          <h3 className="flex items-center gap-2"><Database size={16} className="text-violet"/> ROC Curve</h3>
          <div className="flex-1">
            {curveData.roc_curve.length === 0 ? (
              <div className="flex-col items-center justify-center h-full text-muted" style={{ height: '100%' }}>
                <BarChart2 size={32} style={{ marginBottom: '0.5rem', opacity: 0.4 }}/>
                <span className="text-sm">Run the pipeline to generate curve data</span>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={curveData.roc_curve}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
                  <XAxis dataKey="fpr" type="number" domain={[0, 1]} stroke="var(--text-muted)" tick={{fontSize: 12}} axisLine={false} tickLine={false} />
                  <YAxis dataKey="tpr" type="number" domain={[0, 1]} stroke="var(--text-muted)" tick={{fontSize: 12}} axisLine={false} tickLine={false} />
                  <Tooltip cursor={{ stroke: 'rgba(255,255,255,0.1)' }} contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-default)', borderRadius: '8px' }} />
                  <Line type="monotone" dataKey="tpr" stroke="var(--violet)" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>

      {/* ── Static Result Graphs (PNG) ─────────────────────────────────────── */}
      {graphs.length > 0 && (
        <div className="glass-card p-6 flex-col gap-6 animate-in delay-4">
          <h3 className="flex items-center gap-2">
            <ImageIcon size={16} className="text-amber"/> Result Graphs
          </h3>

          {/* Tab strip */}
          <div className="flex-row gap-2" style={{ flexWrap: 'wrap' }}>
            {graphs.map(g => (
              <button
                key={g}
                id={`graph-tab-${g.replace('.', '-')}`}
                onClick={() => setActiveGraph(g)}
                style={{
                  padding: '0.4rem 1rem',
                  borderRadius: '6px',
                  border: '1px solid var(--border-default)',
                  background: activeGraph === g ? 'var(--violet)' : 'var(--bg-elevated)',
                  color: activeGraph === g ? '#fff' : 'var(--text-secondary)',
                  cursor: 'pointer',
                  fontSize: '0.85rem',
                  fontWeight: activeGraph === g ? '600' : '400',
                  transition: 'all 0.2s ease',
                }}
              >
                {GRAPH_LABELS[g] || g}
              </button>
            ))}
          </div>

          {/* Graph preview */}
          {activeGraph && (
            <div style={{
              background: 'var(--bg-elevated)',
              borderRadius: '12px',
              padding: '1rem',
              border: '1px solid var(--border-subtle)',
              display: 'flex',
              justifyContent: 'center',
            }}>
              <img
                key={curveData.cache_bust}  // forces React to remount img when new model runs
                src={`${API_BASE}/graphs/${activeGraph}${curveData.cache_bust ? `?t=${encodeURIComponent(curveData.cache_bust)}` : ''}`}
                alt={GRAPH_LABELS[activeGraph] || activeGraph}
                style={{ maxWidth: '100%', maxHeight: '500px', borderRadius: '8px', objectFit: 'contain' }}
              />
            </div>
          )}
        </div>
      )}

      {/* CI/CD Pipeline Runs Table */}
      <div className="glass-card p-6 flex-col gap-6 animate-in delay-5">
        <h3 className="flex items-center gap-2"><GitCommit size={16} className="text-emerald"/> Recent Pipeline Runs</h3>
        <div style={{ overflowX: 'auto' }}>
          <table>
            <thead>
              <tr>
                <th>Version</th>
                <th>Model Type</th>
                <th>Status</th>
                <th>PR-AUC</th>
                <th>ROC-AUC</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {evaluations.length === 0 ? (
                <tr><td colSpan={6} className="text-center text-muted" style={{ padding: '2rem' }}>No pipeline runs recorded yet. Run the Prefect flow first.</td></tr>
              ) : evaluations.map((run, idx) => (
                <tr key={idx}>
                  <td className="mono text-violet font-medium">{run.version}</td>
                  <td className="font-medium">{run.model_type}</td>
                  <td><span className="badge badge-emerald"><CheckCircle2 size={12}/> Success</span></td>
                  <td>{(run.auc_pr * 100).toFixed(2)}%</td>
                  <td>{(run.auc_roc * 100).toFixed(2)}%</td>
                  <td className="text-secondary">{new Date(run.timestamp).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
