import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, ChevronRight, Hash, CheckCircle2, XCircle, Clock } from 'lucide-react';
import { pipelineRuns } from '../data/mockData';

const PipelineLog = () => {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');

  const filteredRuns = pipelineRuns.filter(r => 
    r.id.toLowerCase().includes(searchTerm.toLowerCase()) || 
    r.model.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="scroll-container flex-col gap-6">
      <header className="flex-row justify-between items-end animate-in delay-1">
        <div className="flex-col gap-1">
          <span className="label text-violet">CI/CD Triggers</span>
          <h1 className="gradient-text-violet">Pipeline Execution Log</h1>
        </div>
        
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
            {filteredRuns.map((run) => (
              <tr 
                key={run.id} 
                style={{ cursor: 'pointer' }}
                onClick={() => navigate(`/pipeline/${run.id}`)}
              >
                <td className="mono text-violet font-medium">{run.id}</td>
                <td className="font-medium">{run.model}</td>
                <td className="text-secondary mono text-sm">{run.dataset}</td>
                <td>
                  {run.status === 'success' && <span className="badge badge-emerald"><CheckCircle2 size={12}/> Success</span>}
                  {run.status === 'failed' && <span className="badge badge-rose"><XCircle size={12}/> Failed</span>}
                  {run.status === 'running' && <span className="badge badge-amber"><Clock size={12}/> Running</span>}
                </td>
                <td className="text-secondary">{run.author}</td>
                <td className="text-secondary">{run.time}</td>
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
