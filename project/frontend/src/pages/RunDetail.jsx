import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, GitMerge, CheckCircle2, XCircle, Clock, Activity, Cpu, Database } from 'lucide-react';

const RunDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [run, setRun] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8000/model_evaluations')
      .then(res => res.json())
      .then(data => {
         if(data.evaluations) {
           const found = data.evaluations.find(r => r.version === id);
           setRun(found);
         }
      })
      .catch(err => console.error("Failed to load run details", err));
  }, [id]);

  if (!run) return <div className="p-8 text-center text-muted">Loading or Run not found</div>;

  return (
    <div className="scroll-container flex-col gap-8">
      <header className="flex-row items-center gap-4 animate-in delay-1">
        <button className="btn btn-ghost" style={{ padding: '0.5rem' }} onClick={() => navigate('/pipeline')}>
          <ArrowLeft size={20} />
        </button>
        <div className="flex-col">
          <span className="label text-violet">Pipeline Details</span>
          <h1 className="gradient-text-violet">{run.version}</h1>
        </div>
      </header>

      <div className="grid-2 animate-in delay-2">
        {/* Pipeline Graph visualization */}
        <div className="glass-card p-6 flex-col gap-6">
          <h3 className="flex items-center gap-2"><GitMerge size={16} className="text-cyan"/> Execution Graph</h3>
          <div className="flex-col gap-4 p-4" style={{ background: 'rgba(0,0,0,0.2)', borderRadius: '12px', border: '1px solid var(--border-subtle)' }}>
            <div className="flex-row items-center gap-4">
              <div style={{ width: '24px', height: '24px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--emerald-dim)', color: 'var(--emerald)' }}>
                <CheckCircle2 size={14}/>
              </div>
              <div className="flex-1 flex-row justify-between items-center" style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <span className="font-medium">Model Pipeline Trained</span>
              </div>
            </div>
          </div>
        </div>

        <div className="flex-col gap-6">
          {/* Metadata */}
          <div className="glass-card p-6 flex-col gap-6">
            <h3 className="flex items-center gap-2"><Database size={16} className="text-violet"/> Environment</h3>
            <div className="flex-col gap-3">
              <div className="flex-row justify-between">
                <span className="text-muted">Target Model</span>
                <span className="font-medium">{run.model_type}</span>
              </div>
              <div className="flex-row justify-between">
                <span className="text-muted">Hyperparameters</span>
                <span className="mono text-sm text-cyan">{run.hyperparameters}</span>
              </div>
              <div className="flex-row justify-between">
                <span className="text-muted">Time</span>
                <span className="text-secondary">{new Date(run.timestamp).toLocaleString()}</span>
              </div>
            </div>
          </div>

          {/* Metrics */}
          <div className="glass-card p-6 flex-col gap-6">
            <h3 className="flex items-center gap-2"><Cpu size={16} className="text-emerald"/> Result Metrics</h3>
            <div className="grid-2">
              <div className="p-4" style={{ background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <span className="label">ROC-AUC</span>
                <div className="stat-number mt-2 text-emerald">{(run.auc_roc * 100).toFixed(2)}%</div>
              </div>
              <div className="p-4" style={{ background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <span className="label">PR-AUC</span>
                <div className="stat-number mt-2 text-violet">{(run.auc_pr * 100).toFixed(2)}%</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RunDetail;
