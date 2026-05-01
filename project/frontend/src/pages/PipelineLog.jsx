import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, ChevronRight, Hash, CheckCircle2, XCircle, Clock, RefreshCw } from 'lucide-react';

const PipelineLog = () => {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');
  const [runs, setRuns] = useState([]);
  const [pipelineStatus, setPipelineStatus] = useState({ status: 'IDLE' });

  const fetchRuns = () => {
    fetch('http://localhost:8000/model_evaluations')
      .then(res => res.json())
      .then(data => {
         if(data.evaluations) setRuns(data.evaluations.reverse());
      })
      .catch(err => console.error("Failed to load runs", err));

    fetch('http://localhost:8000/pipeline/status')
      .then(res => res.json())
      .then(data => setPipelineStatus(data))
      .catch(err => console.error("Failed to load status", err));
  };

  useEffect(() => {
    fetchRuns();
    const interval = setInterval(fetchRuns, 10000);
    return () => clearInterval(interval);
  }, []);

  const filteredRuns = runs.filter(r => 
    r.version.toLowerCase().includes(searchTerm.toLowerCase()) || 
    r.model_type.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="scroll-container flex-col gap-6">
      <header className="flex-row justify-between items-end animate-in delay-1">
        <div className="flex-col gap-1">
          <span className="label text-violet">CI/CD Triggers</span>
          <h1 className="gradient-text-violet">Pipeline Execution Log</h1>
        </div>

        <div className="flex-row gap-4 items-center">
          {pipelineStatus.status === 'RUNNING' && (
            <button 
              className="btn btn-ghost btn-sm" 
              onClick={async () => {
                if (!window.confirm("Cancel this build?")) return;
                await fetch(`http://localhost:8000/pipeline/cancel/${pipelineStatus.id}`, { method: 'POST' });
                fetchRuns();
              }}
              style={{ borderColor: 'rgba(244, 63, 94, 0.2)', color: 'var(--rose)' }}
            >
              <XCircle size={14} /> Kill Active Build
            </button>
          )}
          <div style={{ width: '300px', position: 'relative' }}>
          <Search size={18} style={{ position: 'absolute', top: '50%', left: '1rem', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
          <input 
            type="text" 
            placeholder="Search run ID or model..." 
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="input"
            style={{ paddingLeft: '2.5rem' }}
          />
          </div>
        </div>
      </header>

      <div className="glass-card animate-in delay-2">
        <table>
          <thead>
            <tr>
              <th>Run ID</th>
              <th>Model Target</th>
              <th>Dataset</th>
              <th>Status</th>
              <th>Author</th>
              <th>Time</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {pipelineStatus.status === 'RUNNING' && (
              <tr style={{ background: 'rgba(245, 158, 11, 0.05)' }}>
                <td className="mono text-amber font-medium">{pipelineStatus.id?.slice(0, 8)}...</td>
                <td className="font-medium text-amber">Building: {pipelineStatus.name}</td>
                <td className="text-secondary mono text-sm">train.csv</td>
                <td>
                  <span className="badge badge-amber" style={{ background: 'rgba(245, 158, 11, 0.1)', color: 'var(--amber)', border: '1px solid rgba(245, 158, 11, 0.2)' }}>
                    <RefreshCw size={12} className="animate-spin" /> {pipelineStatus.state}
                  </span>
                </td>
                <td className="text-secondary">System</td>
                <td className="text-secondary">{new Date(pipelineStatus.start_time).toLocaleString()}</td>
                <td></td>
              </tr>
            )}
            {filteredRuns.map((run) => (
              <tr 
                key={run.version} 
                style={{ cursor: 'pointer' }}
                onClick={() => navigate(`/pipeline/${run.version}`)}
              >
                <td className="mono text-violet font-medium">{run.version}</td>
                <td className="font-medium">{run.model_type}</td>
                <td className="text-secondary mono text-sm">train.csv</td>
                <td>
                  <span className="badge badge-emerald"><CheckCircle2 size={12}/> Success</span>
                </td>
                <td className="text-secondary">System</td>
                <td className="text-secondary">{new Date(run.timestamp).toLocaleString()}</td>
                <td className="text-right">
                  <ChevronRight size={18} className="text-muted" />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {filteredRuns.length === 0 && <div className="p-8 text-center text-muted">No runs found matching "{searchTerm}"</div>}
      </div>
    </div>
  );
};

export default PipelineLog;
