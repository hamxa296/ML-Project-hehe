import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, GitMerge, CheckCircle2, XCircle, Clock, Activity, Cpu, Database } from 'lucide-react';
import { pipelineRuns } from '../data/mockData';

const RunDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const run = pipelineRuns.find(r => r.id === id);

  if (!run) return <div className="p-8 text-center text-muted">Run not found</div>;

  return (
    <div className="scroll-container flex-col gap-8">
      <header className="flex-row items-center gap-4 animate-in delay-1">
        <button className="btn btn-ghost" style={{ padding: '0.5rem' }} onClick={() => navigate('/pipeline')}>
          <ArrowLeft size={20} />
        </button>
        <div className="flex-col">
          <span className="label text-violet">Pipeline Details</span>
          <h1 className="gradient-text-violet">{run.id}</h1>
        </div>
      </header>

      <div className="grid-2 animate-in delay-2">
        {/* Pipeline Graph visualization */}
        <div className="glass-card p-6 flex-col gap-6">
          <h3 className="flex items-center gap-2"><GitMerge size={16} className="text-cyan"/> Execution Graph</h3>
          <div className="flex-col gap-4 p-4" style={{ background: 'rgba(0,0,0,0.2)', borderRadius: '12px', border: '1px solid var(--border-subtle)' }}>
            {run.stages.map((stage, idx) => (
              <div key={idx} className="flex-row items-center gap-4">
                <div style={{ width: '24px', height: '24px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: stage.status === 'done' ? 'var(--emerald-dim)' : stage.status === 'failed' ? 'var(--rose-dim)' : stage.status === 'running' ? 'var(--amber-dim)' : 'rgba(255,255,255,0.05)', color: stage.status === 'done' ? 'var(--emerald)' : stage.status === 'failed' ? 'var(--rose)' : stage.status === 'running' ? 'var(--amber)' : 'var(--text-muted)' }}>
                  {stage.status === 'done' && <CheckCircle2 size={14}/>}
                  {stage.status === 'failed' && <XCircle size={14}/>}
                  {stage.status === 'running' && <Clock size={14}/>}
                  {stage.status === 'pending' && <div style={{ width: '6px', height: '6px', background: 'currentColor', borderRadius: '50%' }}/>}
                </div>
                <div className="flex-1 flex-row justify-between items-center" style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                  <span className="font-medium">{stage.name}</span>
                  {stage.error && <span className="text-rose text-sm bg-rose/10 px-2 py-1 rounded">{stage.error}</span>}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="flex-col gap-6">
          {/* Metadata */}
          <div className="glass-card p-6 flex-col gap-6">
            <h3 className="flex items-center gap-2"><Database size={16} className="text-violet"/> Environment</h3>
            <div className="flex-col gap-3">
              <div className="flex-row justify-between">
                <span className="text-muted">Target Model</span>
                <span className="font-medium">{run.model}</span>
              </div>
              <div className="flex-row justify-between">
                <span className="text-muted">Dataset Target</span>
                <span className="mono text-sm text-cyan">{run.dataset}</span>
              </div>
              <div className="flex-row justify-between">
                <span className="text-muted">Trigger</span>
                <span className="text-secondary">{run.author}</span>
              </div>
            </div>
          </div>

          {/* Metrics */}
          <div className="glass-card p-6 flex-col gap-6">
            <h3 className="flex items-center gap-2"><Cpu size={16} className="text-emerald"/> Result Metrics</h3>
            <div className="grid-2">
              <div className="p-4" style={{ background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <span className="label">Accuracy</span>
                <div className="stat-number mt-2 text-emerald">{run.accuracy ? `${run.accuracy}%` : '--'}</div>
              </div>
              <div className="p-4" style={{ background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <span className="label">F1 Score</span>
                <div className="stat-number mt-2 text-violet">{run.f1Score || '--'}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RunDetail;
