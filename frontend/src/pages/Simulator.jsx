import { useState } from 'react';
import { PlayCircle, Cpu, ShieldCheck, Loader2, AlertCircle } from 'lucide-react';
import { predictTransaction } from '../services/api';

const Simulator = () => {
  const [formData, setFormData] = useState({
    TransactionAmt: 1500,
    card1: 1000,
    addr1: 120,
    TransactionDT: 86400,
    P_emaildomain: 'gmail.com'
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [pred, setPred] = useState(null);

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await predictTransaction(formData);
      
      const probPercent = (result.probability * 100).toFixed(1);
      
      let classLabel, color, glow;
      if (result.is_fraud === 1) {
        classLabel = 'Fraud';
        color = 'var(--rose)';
        glow = 'var(--rose-glow)';
      } else if (result.probability > 0.5) {
        classLabel = 'Suspicious';
        color = 'var(--amber)';
        glow = 'var(--amber-glow)';
      } else {
        classLabel = 'Legitimate';
        color = 'var(--emerald)';
        glow = 'var(--emerald-glow)';
      }

      setPred({ prob: `${probPercent}%`, class: classLabel, color, glow });
      
      // Save to local storage for Dashboard stats
      const history = JSON.parse(localStorage.getItem('predictionHistory') || '[]');
      history.push({ ...result, timestamp: new Date().toISOString() });
      localStorage.setItem('predictionHistory', JSON.stringify(history));
      
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: name === 'P_emaildomain' ? value : Number(value) }));
  };

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
          
          <div className="flex-col gap-4">
            <div className="flex-col gap-2">
              <div className="flex-row justify-between items-center">
                <label className="text-secondary text-sm">Transaction Volume ($)</label>
                <span className="mono font-medium text-violet">${formData.TransactionAmt}</span>
              </div>
              <input 
                name="TransactionAmt"
                type="range" min="0" max="5000" step="50" 
                value={formData.TransactionAmt} onChange={handleChange}
                style={{ width: '100%', accentColor: 'var(--violet)', cursor: 'pointer' }}
              />
            </div>
            
            <div className="flex-col gap-2">
              <label className="text-secondary text-sm">Card ID (card1)</label>
              <input 
                name="card1"
                type="number" className="input" 
                value={formData.card1} onChange={handleChange}
              />
            </div>

            <div className="flex-col gap-2">
              <label className="text-secondary text-sm">Billing Region (addr1)</label>
              <input 
                name="addr1"
                type="number" className="input" 
                value={formData.addr1} onChange={handleChange}
              />
            </div>
            
            <button 
              onClick={handlePredict} 
              disabled={loading}
              className="mt-4"
              style={{
                background: 'var(--violet)', color: 'white', padding: '0.75rem', borderRadius: '8px', 
                border: 'none', cursor: loading ? 'not-allowed' : 'pointer', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.5rem'
              }}
            >
              {loading ? <Loader2 className="animate-spin" size={18} /> : 'Run Prediction'}
            </button>
            
            {error && (
              <div className="p-3 mt-2 rounded-md flex items-center gap-2" style={{ background: 'rgba(244, 63, 94, 0.1)', color: 'var(--rose)', border: '1px solid var(--rose)' }}>
                <AlertCircle size={16} />
                <span className="text-sm">{error}</span>
              </div>
            )}
          </div>
        </div>

        <div className="flex-col gap-6">
          {pred ? (
            <div className="glass-card p-8 flex-col gap-4 justify-center items-center text-center" style={{ flex: 1, borderTop: `4px solid ${pred.color}`, boxShadow: `0 8px 30px ${pred.glow}` }}>
              <Cpu size={32} color={pred.color} />
              <h3 className="mt-2 text-muted">Model Output Prediction</h3>
              <div className="stat-number" style={{ color: pred.color }}>{pred.class}</div>
              <span className="badge" style={{ background: pred.glow, color: pred.color, border: `1px solid ${pred.color}` }}>
                Confidence: {pred.prob}
              </span>
            </div>
          ) : (
            <div className="glass-card p-8 flex-col gap-4 justify-center items-center text-center opacity-50" style={{ flex: 1 }}>
              <Cpu size={32} className="text-muted" />
              <h3 className="mt-2 text-muted">Awaiting Input</h3>
              <p className="text-sm text-secondary">Run prediction to see results.</p>
            </div>
          )}

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
