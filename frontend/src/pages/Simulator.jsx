import { useState } from 'react';
import { PlayCircle, Cpu, ShieldCheck } from 'lucide-react';

const Simulator = () => {
  const [val, setVal] = useState(1500);

  const getPrediction = () => {
    if (val > 2500) return { prob: '98%', class: 'Fraud', color: 'var(--rose)', glow: 'var(--rose-glow)' };
    if (val > 800) return { prob: '65%', class: 'Suspicious', color: 'var(--amber)', glow: 'var(--amber-glow)' };
    return { prob: '12%', class: 'Legitimate', color: 'var(--emerald)', glow: 'var(--emerald-glow)' };
  };

  const pred = getPrediction();

  return (
    <div className="scroll-container flex-col gap-6">
      <header className="flex-row justify-between items-end animate-in delay-1">
        <div className="flex-col gap-1">
          <span className="label text-emerald">Live Inference Sandbox</span>
          <h1 className="gradient-text-violet">Model Simulator</h1>
        </div>
      </header>

      <div className="grid-2 animate-in delay-2">
        <div className="glass-card p-8 flex-col gap-8">
          <h3 className="flex items-center gap-2"><PlayCircle size={18} className="text-violet"/> Test Input Features</h3>
          
          <div className="flex-col gap-2">
            <div className="flex-row justify-between items-center">
              <label className="text-secondary text-sm">Transaction Volume ($)</label>
              <span className="mono font-medium text-violet">${val}</span>
            </div>
            <input 
              type="range" min="0" max="5000" step="50" 
              value={val} onChange={(e) => setVal(Number(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--violet)', cursor: 'pointer' }}
            />
          </div>

          <div className="flex-col gap-4 opacity-50 pointer-events-none">
            <div className="flex-col gap-2">
              <label className="text-secondary text-sm">Time of Day (Locked for simulation)</label>
              <input type="text" className="input" value="03:45 AM" readOnly />
            </div>
            <div className="flex-col gap-2">
              <label className="text-secondary text-sm">Location Match (Locked for simulation)</label>
              <input type="text" className="input" value="Mismatch (IP: RU -> Billing: US)" readOnly />
            </div>
          </div>
        </div>

        <div className="flex-col gap-6">
          <div className="glass-card p-8 flex-col gap-4 justify-center items-center text-center" style={{ flex: 1, borderTop: `4px solid ${pred.color}`, boxShadow: `0 8px 30px ${pred.glow}` }}>
            <Cpu size={32} color={pred.color} />
            <h3 className="mt-2 text-muted">Model Output Prediction</h3>
            <div className="stat-number" style={{ color: pred.color }}>{pred.class}</div>
            <span className="badge" style={{ background: pred.glow, color: pred.color, border: `1px solid ${pred.color}` }}>
              Confidence: {pred.prob}
            </span>
          </div>

          <div className="glass-card p-6 flex-row items-center gap-4">
            <ShieldCheck className="text-cyan" size={24} />
            <div className="flex-col">
              <span className="font-medium">Model Endpoint</span>
              <span className="text-sm text-secondary mono">v2.1-xgboost-prod</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Simulator;
