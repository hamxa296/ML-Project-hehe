import { useState, useEffect } from 'react';
import { Box, Layers, Calendar, HardDrive, CheckCircle2, Play, Info, AlertCircle, RefreshCw } from 'lucide-react';

const API = 'http://localhost:8000';

const SC = ({ title, value, icon: Icon, color = 'cyan' }) => (
  <div className="glass-card p-5 flex-col gap-2" style={{ borderTop: `2px solid var(--${color})` }}>
    <div className="flex-row items-center gap-2">
      <Icon size={14} className={`text-${color}`} />
      <span className="label text-muted">{title}</span>
    </div>
    <span className="stat-number">{value}</span>
  </div>
);

export default function Models() {
  const [models, setModels] = useState([]);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activating, setActivating] = useState(null);
  const [activeModelName, setActiveModelName] = useState('model_latest.pkl');

  const fetchData = async () => {
    try {
      const [mRes, hRes] = await Promise.all([
        fetch(`${API}/models`).then(r => r.json()),
        fetch(`${API}/model_evaluations`).then(r => r.json()),
      ]);
      setModels(mRes.models || []);
      setHistory(hRes.evaluations || []);
    } catch (e) {
      console.error('Failed to fetch models:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleActivate = async (name) => {
    setActivating(name);
    try {
      const res = await fetch(`${API}/models/activate/${name}`, { method: 'POST' });
      if (res.ok) {
        setActiveModelName(name);
        await fetchData(); // Refresh to show latest stats
      }
    } catch (e) {
      alert('Failed to activate model');
    } finally {
      setActivating(null);
    }
  };

  // Find metrics for a specific model version from history
  const getMetrics = (name) => {
    // Filenames are like model_v_20260430_171337.pkl
    // results.csv version is v_20260430_171337
    const versionId = name.replace('model_', '').replace('.pkl', '');
    return history.find(h => h.version === versionId);
  };

  if (loading) return <div className="p-8 text-center text-muted">Loading model registry...</div>;

  return (
    <div className="scroll-container flex-col gap-6">
      <header className="flex-row justify-between items-end animate-in delay-1">
        <div className="flex-col gap-1">
          <span className="label text-violet">Model Management</span>
          <h1 className="gradient-text">Model Registry</h1>
        </div>
        <button className="btn btn-ghost" onClick={() => { setLoading(true); fetchData(); }}>
          <RefreshCw size={15} /> Refresh Registry
        </button>
      </header>

      <div className="grid-3 animate-in delay-2">
        <SC title="Total Versions" value={models.length} icon={Layers} color="cyan" />
        <SC title="Storage Used" value={`${models.reduce((a, b) => a + b.size_mb, 0).toFixed(1)} MB`} icon={HardDrive} color="violet" />
        <SC title="Latest Run" value={models[0]?.created_at ? new Date(models[0].created_at).toLocaleDateString() : 'N/A'} icon={Calendar} color="emerald" />
      </div>

      <div className="flex-col gap-4 animate-in delay-3">
        <div className="flex-row items-center gap-2">
          <Box size={18} className="text-cyan" />
          <h2 style={{ margin: 0, fontSize: '1.1rem' }}>Available Model Artefacts</h2>
        </div>

        <div className="glass-card" style={{ padding: 0, overflow: 'hidden' }}>
          <table className="w-full text-left" style={{ borderCollapse: 'collapse' }}>
            <thead style={{ background: 'rgba(255,255,255,0.02)', borderBottom: '1px solid var(--border)' }}>
              <tr>
                <th className="p-4 label text-muted">Version / Filename</th>
                <th className="p-4 label text-muted">Created</th>
                <th className="p-4 label text-muted">Performance (AUC-PR)</th>
                <th className="p-4 label text-muted">Size</th>
                <th className="p-4 label text-muted text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m, i) => {
                const metrics = getMetrics(m.name);
                const isActive = activeModelName === m.name;
                
                return (
                  <tr key={i} style={{ borderBottom: '1px solid var(--border)', background: isActive ? 'rgba(16, 185, 129, 0.03)' : 'transparent' }}>
                    <td className="p-4">
                      <div className="flex-col">
                        <span style={{ fontWeight: 600, color: isActive ? 'var(--emerald)' : 'var(--text)' }}>
                          {m.name}
                        </span>
                        <span className="text-xs text-muted">
                          {metrics ? 'Verified in History' : 'Unindexed artefact'}
                        </span>
                      </div>
                    </td>
                    <td className="p-4 text-sm text-muted">
                      {new Date(m.created_at).toLocaleString()}
                    </td>
                    <td className="p-4">
                      {metrics ? (
                        <div className="flex-row items-center gap-2">
                          <div style={{ width: 60, height: 4, background: '#30363d', borderRadius: 2 }}>
                            <div style={{ width: `${metrics.auc_pr * 100}%`, height: '100%', background: 'var(--emerald)', borderRadius: 2 }} />
                          </div>
                          <span className="text-sm font-mono">{(metrics.auc_pr * 100).toFixed(1)}%</span>
                        </div>
                      ) : (
                        <span className="text-xs text-muted italic">No metrics data</span>
                      )}
                    </td>
                    <td className="p-4 text-sm text-muted">
                      {m.size_mb} MB
                    </td>
                    <td className="p-4 text-right">
                      {isActive ? (
                        <div className="flex-row items-center gap-1 justify-end text-emerald">
                          <CheckCircle2 size={16} />
                          <span className="label" style={{ color: 'var(--emerald)' }}>Active Production</span>
                        </div>
                      ) : (
                        <button 
                          className="btn btn-ghost btn-sm" 
                          disabled={activating === m.name}
                          onClick={() => handleActivate(m.name)}
                          style={{ borderColor: 'var(--border)' }}
                        >
                          {activating === m.name ? (
                            <RefreshCw size={14} className="animate-spin" />
                          ) : (
                            <Play size={14} />
                          )}
                          Activate
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {models.length === 0 && (
            <div className="p-12 text-center text-muted flex-col items-center gap-3">
              <AlertCircle size={30} opacity={0.3} />
              <p>No versioned models found in /models directory.</p>
            </div>
          )}
        </div>
      </div>

      <div className="glass-card p-6 flex-row gap-4 items-start" style={{ background: 'rgba(56, 189, 248, 0.02)', borderColor: 'rgba(56, 189, 248, 0.2)' }}>
        <Info className="text-cyan" size={20} style={{ marginTop: 2 }} />
        <div className="flex-col gap-1">
          <h4 style={{ margin: 0, color: 'var(--cyan)' }}>About Hot-Swapping</h4>
          <p className="text-sm text-muted m-0">
            Activating a model here performs a <strong>Hot-Reload</strong>. The API swaps the weights in memory instantly without 
            restarting the service. It also updates the persistent pointer so this version remains active after a reboot.
          </p>
        </div>
      </div>
    </div>
  );
}
