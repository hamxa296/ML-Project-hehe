import { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Legend, ScatterChart, Scatter, Cell
} from 'recharts';
import { Database, Activity, Image, ZoomIn, X, ChevronLeft, ChevronRight } from 'lucide-react';

const API = 'http://localhost:8000';
const C = { emerald: '#10b981', rose: '#f43f5e', cyan: '#06b6d4', violet: '#d4d4d4', amber: '#f59e0b' };
const CLUSTER_COLORS = ['#06b6d4', '#d4d4d4', '#f43f5e', '#10b981', '#f59e0b'];

/* ── Tooltip ── */
const CT = ({ active, payload, label }) => active && payload?.length ? (
  <div style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 8, padding: '10px 14px' }}>
    <p style={{ color: '#e6edf3', fontWeight: 700, marginBottom: 4, fontSize: 13 }}>{label}</p>
    {payload.map((e, i) => <p key={i} style={{ color: e.color, margin: 0, fontSize: 12 }}>{e.name}: <b>{typeof e.value === 'number' ? e.value.toFixed(4) : e.value}</b></p>)}
  </div>
) : null;

/* ── Stat Card ── */
const SC = ({ title, value, sub, color = 'cyan' }) => (
  <div className="glass-card p-5 flex-col gap-2" style={{ borderTop: `2px solid var(--${color})` }}>
    <span className="label text-muted">{title}</span>
    <span className="stat-number" style={{ fontSize: '1.8rem' }}>{value}</span>
    {sub && <span className="text-sm text-muted">{sub}</span>}
  </div>
);

/* ── Plot Card (PNG viewer with lightbox) ── */
const PlotCard = ({ src, title, onClick }) => (
  <div className="glass-card flex-col gap-0" style={{ cursor: 'pointer', overflow: 'hidden' }} onClick={onClick}>
    <div style={{ background: '#0d1117', padding: '12px 16px', borderBottom: '1px solid #30363d', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <span style={{ fontSize: 12, fontWeight: 600, color: '#e6edf3' }}>{title}</span>
      <ZoomIn size={14} color="#6b7280" />
    </div>
    <img src={src} alt={title} style={{ width: '100%', display: 'block', objectFit: 'contain', background: '#0d1117' }} loading="lazy" />
  </div>
);

/* ── Lightbox ── */
const Lightbox = ({ images, index, onClose, onNav }) => {
  useEffect(() => {
    const handler = e => { if (e.key === 'Escape') onClose(); if (e.key === 'ArrowRight') onNav(1); if (e.key === 'ArrowLeft') onNav(-1); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose, onNav]);
  if (index === null) return null;
  return (
    <div onClick={onClose} style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.92)', zIndex: 9999, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <button onClick={e => { e.stopPropagation(); onNav(-1); }} style={{ position: 'absolute', left: 24, background: '#161b22', border: '1px solid #30363d', borderRadius: 8, padding: '10px 14px', color: '#e6edf3', cursor: 'pointer' }}><ChevronLeft size={20} /></button>
      <img onClick={e => e.stopPropagation()} src={images[index].src} alt={images[index].title} style={{ maxWidth: '88vw', maxHeight: '88vh', borderRadius: 8, boxShadow: '0 20px 80px rgba(0,0,0,0.8)' }} />
      <button onClick={e => { e.stopPropagation(); onNav(1); }} style={{ position: 'absolute', right: 24, background: '#161b22', border: '1px solid #30363d', borderRadius: 8, padding: '10px 14px', color: '#e6edf3', cursor: 'pointer' }}><ChevronRight size={20} /></button>
      <button onClick={onClose} style={{ position: 'absolute', top: 20, right: 20, background: '#161b22', border: '1px solid #30363d', borderRadius: 8, padding: 8, color: '#e6edf3', cursor: 'pointer' }}><X size={18} /></button>
      <div style={{ position: 'absolute', bottom: 20, color: '#6b7280', fontSize: 13 }}>{images[index].title} ({index + 1}/{images.length})</div>
    </div>
  );
};

const PLOT_LABELS = {
  '01_target_distribution.png': 'Class Imbalance',
  '02_missing_values.png': 'Missing Values',
  '03_transaction_amount.png': 'Transaction Amount',
  '04_time_analysis.png': 'Time-Based Fraud Patterns',
  '05_c_features.png': 'Count Features (C)',
  '06_d_features.png': 'Timedelta Features (D)',
  '07_v_features_corr.png': 'Top V-Feature Correlations',
  '08_v_features_missing.png': 'V-Feature Missingness',
  '09_correlation_matrix.png': 'Correlation Matrix',
  '10_category_fraud_rates.png': 'Category Fraud Rates',
  '11_outlier_analysis.png': 'Outlier Analysis',
  'p01_engineered_features.png': 'Engineered Features',
  'p02_cluster_analysis.png': 'K-Means Cluster Overview',
  'p03_cluster_amounts.png': 'Transaction Amount per Cluster',
  'p04_pca_visualisation.png': 'PCA 2D Projection',
  'p05_top_feature_distributions.png': 'Top Feature Distributions',
  'p06_processed_correlation.png': 'Processed Correlation Heatmap',
  'p07_mutual_information.png': 'Mutual Information Scores',
};

const PRIORITY_RAW    = ['01_target_distribution.png', '04_time_analysis.png', '07_v_features_corr.png', '09_correlation_matrix.png', '10_category_fraud_rates.png', '11_outlier_analysis.png', '03_transaction_amount.png', '02_missing_values.png', '05_c_features.png', '06_d_features.png', '08_v_features_missing.png'];
const PRIORITY_PROC   = ['p02_cluster_analysis.png', 'p03_cluster_amounts.png', 'p04_pca_visualisation.png', 'p01_engineered_features.png', 'p07_mutual_information.png', 'p05_top_feature_distributions.png', 'p06_processed_correlation.png'];

export default function EDA() {
  const [tab, setTab]         = useState('raw');
  const [rawData, setRawData] = useState(null);
  const [procData, setProcData] = useState(null);
  const [plots, setPlots]     = useState({ raw: [], processed: [] });
  const [loading, setLoading] = useState(true);
  const [lbIdx, setLbIdx]     = useState(null);
  const [lbCat, setLbCat]     = useState('raw');

  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const [rj, pj, pl] = await Promise.all([
          fetch(`${API}/eda_stats/raw`).then(r => r.json()),
          fetch(`${API}/eda_stats/processed`).then(r => r.json()),
          fetch(`${API}/eda_list`).then(r => r.json()),
        ]);
        if (rj?.summary) setRawData(rj);
        if (pj?.summary) setProcData(pj);
        if (pl?.raw)     setPlots(pl);
      } catch (e) { console.error(e); }
      setLoading(false);
    })();
  }, []);

  const makePlotImgs = (cat, names) =>
    names.map(n => ({ src: `${API}/eda/${cat}/${n}`, title: PLOT_LABELS[n] || n }));

  const rawImgs  = makePlotImgs('raw',       PRIORITY_RAW.filter(n => plots.raw?.includes(n)));
  const procImgs = makePlotImgs('processed', PRIORITY_PROC.filter(n => plots.processed?.includes(n)));
  const curImgs  = lbCat === 'raw' ? rawImgs : procImgs;

  const openLb = (cat, idx) => { setLbCat(cat); setLbIdx(idx); };
  const navLb  = d => setLbIdx(i => (i + d + curImgs.length) % curImgs.length);

  if (loading) return <div className="p-8 text-center text-muted">Loading EDA data...</div>;

  const noData = (msg) => (
    <div className="glass-card p-8 text-center" style={{ color: '#6b7280' }}>
      <Database size={40} style={{ margin: '0 auto 12px', display: 'block', opacity: 0.3 }} />
      <p className="m-0">{msg || 'Run the pipeline to generate EDA.'}</p>
    </div>
  );

  return (
    <div className="scroll-container flex-col gap-6">
      <Lightbox images={curImgs} index={lbIdx} onClose={() => setLbIdx(null)} onNav={navLb} />

      {/* ── Header ── */}
      <header className="flex-row justify-between items-end animate-in delay-1">
        <div className="flex-col gap-1">
          <span className="label text-cyan">Exploratory Data Analysis</span>
          <h1 className="gradient-text">Dataset Profiling</h1>
        </div>
        <div className="flex-row gap-2">
          {[['raw', <Database size={15}/>, 'Raw Data'], ['proc', <Activity size={15}/>, 'Processed Data']].map(([id, icon, label]) => (
            <button key={id} className={`btn ${tab === id ? 'btn-primary' : 'btn-ghost'}`} onClick={() => setTab(id)}>
              {icon} {label}
            </button>
          ))}
        </div>
      </header>

      {/* ══════════════ RAW TAB ══════════════ */}
      {tab === 'raw' && (
        <div className="flex-col gap-6 animate-in delay-2">
          {!rawData ? noData() : <>
            {/* Stat row */}
            <div className="grid-4">
              <SC title="Total Rows"     value={rawData.summary.total_rows.toLocaleString()} color="violet" />
              <SC title="Raw Features"   value={rawData.summary.total_features} color="cyan" />
              <SC title="Fraud Rate"     value={`${rawData.summary.fraud_rate.toFixed(2)}%`} color="rose" sub={`${rawData.summary.imbalance_ratio}:1 imbalance`} />
              <SC title="Redundant Pairs" value={rawData.summary.high_corr_pairs} color="amber" sub="r > 0.98 (pruned)" />
            </div>

            {/* Interactive charts row */}
            <div className="grid-2">
              <div className="glass-card p-5 flex-col gap-3" style={{ height: 340 }}>
                <h3 style={{ margin: 0, fontSize: 13 }}>Fraud Rate by Hour of Day</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={rawData.hourly_fraud} margin={{ bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis dataKey="hour" stroke="#555" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#555" tick={{ fontSize: 11 }} />
                    <Tooltip content={<CT />} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Bar dataKey="fraud_rate" name="Fraud Rate (%)" fill={C.rose} radius={[3,3,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="glass-card p-5 flex-col gap-3" style={{ height: 340 }}>
                <h3 style={{ margin: 0, fontSize: 13 }}>Top 10 Feature Correlations with isFraud</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={rawData.top_feature_correlations?.slice(0, 10)} layout="vertical" margin={{ left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis type="number" stroke="#555" tick={{ fontSize: 11 }} />
                    <YAxis dataKey="feature" type="category" width={55} stroke="#555" tick={{ fontSize: 10 }} />
                    <Tooltip content={<CT />} />
                    <Bar dataKey="correlation" name="Pearson |r|" radius={[0,3,3,0]}>
                      {rawData.top_feature_correlations?.slice(0, 10).map((_, i) => (
                        <Cell key={i} fill={`hsl(${185 + i * 8}, 80%, 55%)`} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Amount stats row */}
            <div className="glass-card p-5 flex-col gap-3" style={{ height: 300 }}>
              <h3 style={{ margin: 0, fontSize: 13 }}>Transaction Amount (log-scale) — Safe vs Fraud</h3>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={rawData.amt_histogram?.slice(0, 50)} margin={{ bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis dataKey="bin" stroke="#555" tick={{ fontSize: 9 }} tickFormatter={v => v.toFixed(1)} />
                  <YAxis stroke="#555" tick={{ fontSize: 10 }} />
                  <Tooltip content={<CT />} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="safe" name="Safe" fill={C.emerald} opacity={0.75} />
                  <Bar dataKey="fraud" name="Fraud" fill={C.rose} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* PNG Gallery */}
            <div className="flex-col gap-3">
              <div className="flex-row items-center gap-2">
                <Image size={15} color="#06b6d4" />
                <h3 style={{ margin: 0, fontSize: 13 }}>All Raw EDA Plots <span style={{ color: '#555', fontWeight: 400 }}>— click to enlarge</span></h3>
              </div>
              {rawImgs.length === 0
                ? noData('No plots found — run the pipeline first.')
                : <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(420px, 1fr))', gap: '1rem' }}>
                    {rawImgs.map((img, i) => <PlotCard key={i} src={img.src} title={img.title} onClick={() => openLb('raw', i)} />)}
                  </div>
              }
            </div>
          </>}
        </div>
      )}

      {/* ══════════════ PROCESSED TAB ══════════════ */}
      {tab === 'proc' && (
        <div className="flex-col gap-6 animate-in delay-2">
          {!procData ? noData() : <>
            {/* Stat row */}
            <div className="grid-4">
              <SC title="Processed Features" value={procData.summary.processed_features} color="cyan" sub={`↓ ${procData.summary.features_removed} removed`} />
              <SC title="K-Means Clusters"   value={procData.summary.n_clusters} color="violet" sub="Behavioural segments" />
              <SC title="Missing After Proc"  value={procData.summary.missing_values} color="emerald" sub="Should be 0" />
              <SC title="PC1 Variance"        value={`${procData.summary.pca_pc1_variance}%`} color="amber" />
            </div>

            {/* ── K-Means Section ── */}
            <div className="flex-col gap-3">
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <div style={{ width: 4, height: 20, background: 'var(--violet)', borderRadius: 4 }} />
                <h2 style={{ margin: 0, fontSize: 15 }}>K-Means Cluster Analysis</h2>
              </div>

              <div className="grid-2" style={{ gap: '1rem' }}>
                {/* Cluster size bar */}
                <div className="glass-card p-5 flex-col gap-3" style={{ height: 300 }}>
                  <h3 style={{ margin: 0, fontSize: 13 }}>Cluster Size Distribution</h3>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={procData.cluster_analysis}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                      <XAxis dataKey="cluster" stroke="#555" tickFormatter={v => `C${v}`} tick={{ fontSize: 11 }} />
                      <YAxis stroke="#555" tick={{ fontSize: 10 }} tickFormatter={v => `${(v/1000).toFixed(0)}k`} />
                      <Tooltip content={<CT />} />
                      <Bar dataKey="total" name="Total Samples" radius={[4,4,0,0]}>
                        {procData.cluster_analysis.map((_, i) => <Cell key={i} fill={CLUSTER_COLORS[i % 5]} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Fraud rate per cluster */}
                <div className="glass-card p-5 flex-col gap-3" style={{ height: 300 }}>
                  <h3 style={{ margin: 0, fontSize: 13 }}>Fraud Rate (%) per Cluster</h3>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={procData.cluster_analysis}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                      <XAxis dataKey="cluster" stroke="#555" tickFormatter={v => `C${v}`} tick={{ fontSize: 11 }} />
                      <YAxis stroke="#555" tick={{ fontSize: 10 }} unit="%" />
                      <Tooltip content={<CT />} />
                      <Bar dataKey="fraud_rate" name="Fraud Rate (%)" radius={[4,4,0,0]}>
                        {procData.cluster_analysis.map((_, i) => <Cell key={i} fill={CLUSTER_COLORS[i % 5]} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Stacked safe vs fraud */}
                <div className="glass-card p-5 flex-col gap-3" style={{ height: 300 }}>
                  <h3 style={{ margin: 0, fontSize: 13 }}>Safe vs Fraud per Cluster</h3>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={procData.cluster_analysis}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                      <XAxis dataKey="cluster" stroke="#555" tickFormatter={v => `C${v}`} tick={{ fontSize: 11 }} />
                      <YAxis stroke="#555" tick={{ fontSize: 10 }} tickFormatter={v => `${(v/1000).toFixed(0)}k`} />
                      <Tooltip content={<CT />} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Bar dataKey="safe_count"  name="Safe"  stackId="a" fill={C.emerald} opacity={0.7} />
                      <Bar dataKey="fraud_count" name="Fraud" stackId="a" fill={C.rose} radius={[4,4,0,0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* PCA scatter */}
                {procData.pca_scatter?.length > 0 && (
                  <div className="glass-card p-5 flex-col gap-3" style={{ height: 300 }}>
                    <h3 style={{ margin: 0, fontSize: 13 }}>PCA 2D Projection — Class Separation</h3>
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                        <XAxis type="number" dataKey="pc1" name="PC1" stroke="#555" tick={{ fontSize: 10 }} />
                        <YAxis type="number" dataKey="pc2" name="PC2" stroke="#555" tick={{ fontSize: 10 }} />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} content={<CT />} />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        <Scatter name="Safe"  data={procData.pca_scatter.filter(d => d.label === 0)} fill={C.emerald} fillOpacity={0.3} />
                        <Scatter name="Fraud" data={procData.pca_scatter.filter(d => d.label === 1)} fill={C.rose} fillOpacity={0.9} />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            </div>

            {/* Mutual Information */}
            <div className="glass-card p-5 flex-col gap-3" style={{ height: 340 }}>
              <h3 style={{ margin: 0, fontSize: 13 }}>Mutual Information — Top 20 Processed Features</h3>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={procData.mutual_information?.slice(0, 20)} layout="vertical" margin={{ left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis type="number" stroke="#555" tick={{ fontSize: 10 }} />
                  <YAxis dataKey="feature" type="category" width={80} stroke="#555" tick={{ fontSize: 9 }} />
                  <Tooltip content={<CT />} />
                  <Bar dataKey="mi_score" name="MI Score" radius={[0,3,3,0]}>
                    {procData.mutual_information?.slice(0, 20).map((_, i) => (
                      <Cell key={i} fill={`hsl(${260 + i * 5}, 70%, 60%)`} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* PNG Gallery */}
            <div className="flex-col gap-3">
              <div className="flex-row items-center gap-2">
                <Image size={15} color="#06b6d4" />
                <h3 style={{ margin: 0, fontSize: 13 }}>All Processed EDA Plots <span style={{ color: '#555', fontWeight: 400 }}>— click to enlarge</span></h3>
              </div>
              {procImgs.length === 0
                ? noData('No plots found — run the pipeline first.')
                : <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(420px, 1fr))', gap: '1rem' }}>
                    {procImgs.map((img, i) => <PlotCard key={i} src={img.src} title={img.title} onClick={() => openLb('processed', i)} />)}
                  </div>
              }
            </div>
          </>}
        </div>
      )}
    </div>
  );
}
