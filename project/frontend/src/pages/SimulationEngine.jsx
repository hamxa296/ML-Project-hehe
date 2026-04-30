import { useState, useMemo } from 'react';
import { SlidersHorizontal, TrendingUp, ShieldCheck, Fingerprint } from 'lucide-react';

const SimulationEngine = () => {
  const [simulatedAmount, setSimulatedAmount] = useState(450);
  const [activeDecision, setActiveDecision] = useState(null);

  const assessment = useMemo(() => {
    const amount = simulatedAmount;
    let level, color, score, action, impact;

    if (amount > 1500) {
      level = 'SEVERE';
      color = 'var(--severe)';
      score = 92;
      action = "Block Immediately";
      impact = amount * 1.2; 
    } else if (amount > 500) {
      level = 'SUSPICIOUS';
      color = 'var(--suspicious)';
      score = 64;
      action = "Verify User Identity";
      impact = amount * 0.5;
    } else {
      level = 'SAFE';
      color = 'var(--safe)';
      score = 12;
      action = "Allow Transaction";
      impact = 0;
    }
    return { level, color, score, action, impact };
  }, [simulatedAmount]);

  return (
    <div className="flex-col gap-8 animate-slide-up" style={{ paddingBottom: '3rem' }}>
      <header className="flex-col stagger-1">
        <span className="section-title"><SlidersHorizontal size={16}/> Interactive "What-if" Simulation</span>
        <h1 className="text-gradient-animate">Simulation Engine</h1>
        <p className="text-secondary" style={{ maxWidth: '600px', marginTop: '0.5rem' }}>
          Manipulate network variables in real-time to observe the decision matrix boundary responses safely without modifying production endpoints.
        </p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
        
        {/* The Core Simulator */}
        <div className="glass-card p-8 flex-col gap-8 stagger-2">
           <div className="flex-row justify-between items-end">
              <div className="flex-col">
                <span className="text-muted" style={{ fontSize: '0.8rem', letterSpacing: '0.1em' }}>ADJUST TARGET VOLUME</span>
                <span style={{ fontSize: '3rem', fontWeight: '800', color: 'var(--accent)' }}>${simulatedAmount.toLocaleString()}</span>
              </div>
           </div>
           
           <input 
             type="range" min="0" max="3000" step="50" 
             value={simulatedAmount} 
             onChange={(e) => setSimulatedAmount(parseInt(e.target.value))}
             style={{ width: '100%', height: '8px', cursor: 'pointer', accentColor: 'var(--accent)', background: 'rgba(255,255,255,0.1)', borderRadius: '4px' }}
           />

           {/* Feature 1: Gamified Risk Score */}
           <div style={{ padding: '1.5rem', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', borderLeft: `6px solid ${assessment.color}` }}>
             <div className="flex-row justify-between items-center mb-4">
                <span style={{ fontWeight: '600', color: 'var(--text-secondary)' }}>Threat Level</span>
                <span style={{ color: assessment.color, fontWeight: '900', fontSize: '1.3rem', textTransform: 'uppercase' }}>
                   {assessment.level}
                </span>
             </div>
             
             <div style={{ position: 'relative', height: '14px', background: 'rgba(255,255,255,0.05)', borderRadius: '10px', overflow: 'hidden' }}>
                <div style={{ 
                  width: `${assessment.score}%`, 
                  height: '100%', 
                  background: assessment.color, 
                  transition: 'all 0.6s cubic-bezier(0.16, 1, 0.3, 1)',
                  boxShadow: `0 0 20px ${assessment.color}`
                }} />
             </div>
           </div>
        </div>

        {/* Feature 3 & 5: Impact and Decisions */}
        <div className="flex-col gap-6 stagger-3">
          
          <div className="glass-card p-8 flex-col gap-4" style={{ flex: 1, justifyContent: 'center' }}>
             <span className="section-title"><TrendingUp size={16}/> Financial Impact Prediction</span>
             <div style={{ fontSize: '3rem', fontWeight: '800', color: 'white' }}>${assessment.impact.toLocaleString()}</div>
             <p className="text-secondary" style={{ fontSize: '0.9rem', margin: 0 }}>Estimated potential maximum loss exposure.</p>
             <span className="badge" style={{ alignSelf: 'flex-start', marginTop: '0.5rem', background: assessment.score > 50 ? 'rgba(166, 93, 93, 0.2)' : 'rgba(140, 166, 126, 0.2)', color: assessment.score > 50 ? 'var(--severe)' : 'var(--safe)' }}>
                {assessment.score > 50 ? 'HIGH PRIORITY' : 'LOW RISK'}
             </span>
          </div>

          <div className="glass-card p-8 flex-col gap-6">
            <h3 className="section-title"><ShieldCheck size={16}/> Smart Decision Panel</h3>
            <div className="flex-col gap-4">
               <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: `1px dashed ${assessment.color}`, color: assessment.color, textAlign: 'center', fontWeight: '700' }}>
                  Recommended: {assessment.action}
               </div>
               <div className="flex-row gap-3">
                  {['Approve', 'Review', 'Block'].map(action => (
                    <button 
                      key={action}
                      onClick={() => setActiveDecision(action)}
                      className="btn-action" 
                      style={{ 
                        flex: 1, padding: '1rem', fontSize: '0.85rem', 
                        background: activeDecision === action ? 'var(--accent)' : 'transparent',
                        color: activeDecision === action ? '#000' : 'var(--accent)'
                      }}
                    >
                       {action}
                    </button>
                  ))}
               </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};

export default SimulationEngine;
