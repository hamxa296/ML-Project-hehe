import { useState, useMemo, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ShieldCheck, AlertTriangle, AlertCircle, DollarSign, BrainCircuit, Activity, SlidersHorizontal, ArrowLeft } from 'lucide-react';
import { mockTransactions } from '../data/mockData';

// Subcomponents pulled from original design
const RiskMeter = ({ riskScore, level }) => {
  const getColor = () => {
    if (level === 'safe') return 'var(--safe)';
    if (level === 'suspicious') return 'var(--suspicious)';
    return 'var(--severe)';
  };
  
  // Animation for the bar width
  const [width, setWidth] = useState(0);
  useEffect(() => {
    setWidth(0);
    const t = setTimeout(() => setWidth(riskScore), 100);
    return () => clearTimeout(t);
  }, [riskScore]);

  return (
    <div className="glass-card p-8 flex-col gap-6" style={{ border: `1px solid ${getColor()}40` }}>
      <div className="flex-row justify-between items-center">
        <h3 className="section-title flex-row items-center gap-2"><Activity size={18} /> Threat Matrix</h3>
        <span style={{ color: getColor(), fontWeight: 'bold', fontSize: '1.2rem', textTransform: 'uppercase', filter: `drop-shadow(0 0 8px ${getColor()})` }}>{level}</span>
      </div>
      <div style={{ height: '14px', background: 'rgba(255,255,255,0.05)', borderRadius: '10px', overflow: 'hidden', boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.5)' }}>
        <div style={{ width: `${width}%`, background: getColor(), height: '100%', transition: 'all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)', boxShadow: `0 0 20px ${getColor()}` }} />
      </div>
      <div className="flex-row justify-between text-muted" style={{ fontSize: '0.85rem', fontWeight: 600 }}>
        <span>Confidence High</span>
        <span>{riskScore}% Intensity</span>
      </div>
    </div>
  );
};

const FraudReasoning = ({ reasons }) => (
  <div className="glass-card p-8 flex-col gap-6" style={{ height: '100%' }}>
    <h3 className="section-title flex-row items-center gap-2"><BrainCircuit size={18} color="var(--accent)"/> Behavioral Narrative</h3>
    <div className="flex-col gap-4">
      {reasons.map((r, i) => (
        <div key={i} className="flex-row gap-4" style={{ padding: '1.2rem', borderLeft: '4px solid var(--accent)', background: 'rgba(197, 160, 89, 0.03)', borderRadius: '0 12px 12px 0', transition: 'transform 0.3s', cursor: 'default' }} onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-2px)'} onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}>
          <span style={{ color: 'white', fontSize: '1rem', fontWeight: '500', lineHeight: '1.5' }}>{r}</span>
        </div>
      ))}
    </div>
  </div>
);

const FinancialImpact = ({ amount, confidence, uncertainty, level }) => (
  <div className="flex-row gap-6">
    <div className="glass-card p-8 flex-col gap-2" style={{ flex: 1.5, borderTop: level === 'severe' ? '4px solid var(--severe)' : level === 'suspicious' ? '4px solid var(--suspicious)' : '4px solid var(--safe)' }}>
      <h3 className="section-title"><DollarSign size={16} /> Exposure</h3>
      <div style={{ fontSize: '3rem', fontWeight: '800', filter: 'drop-shadow(0 0 10px rgba(255,255,255,0.2))' }}>${amount.toLocaleString()}</div>
      <p className="text-secondary" style={{ fontSize: '0.9rem' }}>Calculated potential loss if not mitigated.</p>
    </div>
    <div className="glass-card p-8 flex-col gap-4 justify-center" style={{ flex: 1 }}>
      <div className="flex-col">
        <span className="text-secondary" style={{ fontSize: '0.8rem', fontWeight: '700', letterSpacing: '0.05em' }}>PRECISION</span>
        <span style={{ color: 'var(--safe)', fontWeight: '800', fontSize: '1.2rem', filter: 'drop-shadow(0 0 5px var(--safe-glow))' }}>{confidence}</span>
      </div>
      <div className="flex-col">
        <span className="text-secondary" style={{ fontSize: '0.8rem', fontWeight: '700', letterSpacing: '0.05em' }}>RISK VAR</span>
        <span style={{ color: 'var(--accent)', fontWeight: '800', fontSize: '1.2rem', filter: 'drop-shadow(0 0 5px var(--accent-glow))' }}>{uncertainty}</span>
      </div>
    </div>
  </div>
);

const ActionPanel = ({ status, setStatus }) => (
  <div className="glass-card p-8 flex-col gap-6">
    <h3 className="section-title">Security Protocol</h3>
    <div className="flex-row gap-4">
      {[
        { label: 'Approve', icon: <ShieldCheck size={18}/>, color: 'var(--safe)', action: 'approved', glow: 'var(--safe-glow)' },
        { label: 'Escalate', icon: <AlertCircle size={18}/>, color: 'var(--suspicious)', action: 'reviewing', glow: 'var(--suspicious-glow)' },
        { label: 'Terminate', icon: <AlertTriangle size={18}/>, color: 'var(--severe)', action: 'blocked', glow: 'var(--severe-glow)' }
      ].map(btn => (
        <button 
          key={btn.label}
          className="flex-col items-center justify-center gap-2" 
          style={{ 
            flex: 1, padding: '1.5rem', borderRadius: '8px', background: status === btn.action ? `rgba(${btn.color === 'var(--safe)' ? '140,166,126' : btn.color === 'var(--severe)' ? '166,93,93' : '197,160,89'}, 0.1)` : 'rgba(255,255,255,0.01)', 
            border: status === btn.action ? `1.5px solid ${btn.color}` : '1px solid var(--border-glass-bright)',
            color: status === btn.action ? btn.color : 'var(--text-secondary)',
            transition: 'all 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
            cursor: 'pointer',
            boxShadow: status === btn.action ? `0 0 20px ${btn.glow}` : 'none',
            transform: status === btn.action ? 'scale(1.02)' : 'scale(1)'
          }}
          onClick={() => setStatus(btn.action)}
          onMouseEnter={(e) => { if(status !== btn.action) { e.currentTarget.style.borderColor = btn.color; e.currentTarget.style.color = btn.color; e.currentTarget.style.transform = 'translateY(-2px)'; } }}
          onMouseLeave={(e) => { if(status !== btn.action) { e.currentTarget.style.borderColor = 'var(--border-glass)'; e.currentTarget.style.color = 'var(--text-secondary)'; e.currentTarget.style.transform = 'translateY(0)'; } }}
        >
          {btn.icon}
          <span style={{ fontSize: '0.85rem', fontWeight: '800', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{btn.label}</span>
        </button>
      ))}
    </div>
  </div>
);

const InvestigationView = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const activeTxnBaseline = mockTransactions.find(t => t.id === id);
  
  const [simulatedAmount, setSimulatedAmount] = useState(null);
  const [status, setStatus] = useState(activeTxnBaseline?.status || 'pending');

  const activeTxn = useMemo(() => {
    if (!activeTxnBaseline) return null;
    if (simulatedAmount === null) return activeTxnBaseline;
    let newScore = activeTxnBaseline.riskScore;
    let newLevel = activeTxnBaseline.riskLevel;
    let newAmt = parseFloat(simulatedAmount);
    if (newAmt > 1500) { newScore = Math.max(newScore, 85); newLevel = 'severe'; }
    else if (newAmt > 300) { newScore = 60; newLevel = 'suspicious'; }
    else { newScore = Math.min(newScore, 20); newLevel = 'safe'; }
    return { ...activeTxnBaseline, amount: newAmt, riskScore: newScore, riskLevel: newLevel, financialImpact: newAmt };
  }, [activeTxnBaseline, simulatedAmount]);

  if (!activeTxn) return <div className="p-12 text-center text-secondary">Reference Not Found</div>;

  return (
    <div className="flex-col gap-8 animate-slide-up" style={{ height: '100%', paddingBottom: '2rem' }}>
      <header className="flex-row items-center gap-6 stagger-1">
         <button className="glass-card" onClick={() => navigate('/transactions')} style={{ background: 'transparent', width: '52px', height: '52px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', transition: 'transform 0.3s, background 0.3s' }} onMouseEnter={(e) => { e.currentTarget.style.transform = 'scale(1.05) translateX(-2px)'; e.currentTarget.style.background = 'rgba(255,255,255,0.05)'; }} onMouseLeave={(e) => { e.currentTarget.style.transform = 'scale(1) translateX(0)'; e.currentTarget.style.background = 'transparent'; }}>
            <ArrowLeft size={24} />
         </button>
         <div>
            <span className="section-title">Case Study / <span style={{ color: 'var(--text-secondary)', marginLeft: '0.5rem', fontFamily: 'monospace' }}>{activeTxn.id}</span></span>
            <h1 style={{ margin: 0 }} className="text-gradient-animate">{activeTxn.merchant}</h1>
         </div>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '2.5rem', alignItems: 'stretch' }}>
        <div className="flex-col gap-6 stagger-2">
          <RiskMeter riskScore={activeTxn.riskScore} level={activeTxn.riskLevel} />
          <FinancialImpact amount={activeTxn.financialImpact} confidence={activeTxn.confidence} uncertainty={activeTxn.uncertainty} level={activeTxn.riskLevel} />
          <ActionPanel status={status} setStatus={setStatus} />
        </div>

        <div className="flex-col gap-6 stagger-3">
          <FraudReasoning reasons={activeTxn.reasons} />
          
          <div className="glass-card p-8 flex-col gap-6">
            <h3 className="section-title flex-row items-center gap-2"><SlidersHorizontal size={18} color="var(--accent)"/> What-if Simulation</h3>
            <div className="flex-col gap-6" style={{ padding: '1rem 0' }}>
              <div className="flex-row justify-between items-end">
                <div className="flex-col">
                  <span className="text-secondary" style={{ fontSize: '0.75rem', letterSpacing: '0.1em', fontWeight: 600 }}>SIMULATED VOLUME</span>
                  <span style={{ fontSize: '2.5rem', fontWeight: '800', color: 'var(--accent)', filter: 'drop-shadow(0 0 10px var(--accent-glow))' }}>${parseFloat(simulatedAmount === null ? activeTxnBaseline.amount : simulatedAmount).toLocaleString()}</span>
                </div>
                {simulatedAmount !== null && (
                  <button onClick={() => setSimulatedAmount(null)} className="btn-action" style={{ padding: '0.6rem 1.2rem', fontSize: '0.85rem', background: 'rgba(255,255,255,0.05)', color: 'white', border: '1px solid var(--border-glass)', boxShadow: 'none' }}>Reset Node</button>
                )}
              </div>
              <input type="range" min="0" max="5000" step="50" value={simulatedAmount === null ? activeTxnBaseline.amount : simulatedAmount} onChange={(e) => setSimulatedAmount(e.target.value)} style={{ width: '100%', height: '8px', borderRadius: '4px', background: 'rgba(255,255,255,0.1)', outline: 'none', WebkitAppearance: 'none', cursor: 'pointer' }}/>
            </div>
            <p className="text-secondary" style={{ fontSize: '0.9rem', lineHeight: '1.6' }}>
              Alter the input volume to observe the decision matrix response in real-time. This modulates threat intensity and exposure ratings mathematically without hitting production endpoints.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InvestigationView;
