import { useState, useEffect, useCallback } from 'react';
import { RefreshCw, XCircle } from 'lucide-react';

const API_BASE = import.meta.env.DEV ? 'http://localhost:8000' : '';

const PipelineBanner = () => {
  const [pipelineStatus, setPipelineStatus] = useState({ status: 'IDLE' });

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/pipeline/status`);
      if (res.ok) {
        const data = await res.json();
        setPipelineStatus(data);
      }
    } catch (err) {
      console.error('Failed to fetch pipeline status:', err);
    }
  }, []);

  const handleCancel = async (runId) => {
    if (!window.confirm("Are you sure you want to terminate this training run?")) return;
    try {
      const res = await fetch(`${API_BASE}/pipeline/cancel/${runId}`, { method: 'POST' });
      if (res.ok) {
        setPipelineStatus({ status: 'IDLE' });
        fetchStatus();
      }
    } catch (e) {
      console.error("Cancellation failed", e);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 10000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  if (pipelineStatus.status !== 'RUNNING') return null;

  return (
    <div 
      className="glass-card flex-row items-center gap-5 p-4 animate-in" 
      style={{ 
        position: 'fixed',
        bottom: '30px',
        right: '30px',
        width: 'auto',
        minWidth: '400px',
        zIndex: 1000,
        background: 'rgba(245, 158, 11, 0.08)', 
        borderColor: 'rgba(245, 158, 11, 0.3)', 
        borderLeft: '4px solid var(--amber)',
        backdropFilter: 'blur(16px)',
        boxShadow: '0 12px 40px rgba(0,0,0,0.5)',
        pointerEvents: 'auto'
      }}
    >
      <div className="flex-col items-center justify-center" style={{ 
        width: 36, height: 36, borderRadius: '50%', background: 'rgba(245, 158, 11, 0.1)' 
      }}>
        <RefreshCw size={18} className="text-amber" style={{ animation: 'spin 2s linear infinite' }} />
      </div>
      <div className="flex-col gap-0.5 flex-1">
        <div className="flex-row items-center gap-2">
          <span className="label text-amber" style={{ fontSize: '0.6rem' }}>Active Pipeline Run</span>
          <span className="text-xs text-muted" style={{ fontWeight: 400, fontSize: '0.7rem' }}>
            • Started {new Date(pipelineStatus.start_time).toLocaleTimeString()}
          </span>
        </div>
        <h4 style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-primary)', fontWeight: 600 }}>
          {pipelineStatus.state}: <span style={{ fontWeight: 400 }}>{pipelineStatus.name}</span>
        </h4>
      </div>
      <div className="flex-row gap-3">
         <button 
           className="btn btn-ghost btn-sm" 
           onClick={() => handleCancel(pipelineStatus.id)}
           style={{ 
             borderColor: 'rgba(244, 63, 94, 0.2)', 
             color: 'var(--rose)',
             padding: '0.3rem 0.8rem',
             fontSize: '0.75rem'
           }}
         >
           <XCircle size={14} /> Cancel Build
         </button>
         <div className="badge badge-amber" style={{ background: 'rgba(245, 158, 11, 0.1)', color: 'var(--amber)' }}>
           Live
         </div>
      </div>
    </div>
  );
};

export default PipelineBanner;
