import { useNavigate } from 'react-router-dom';
import { Play, Database, Server, GitMerge, Activity, CheckCircle2 } from 'lucide-react';

const Landing = () => {
  const navigate = useNavigate();

  return (
    <div className="scroll-container p-0 w-full h-screen relative">
      <div className="app-bg" />
      <div className="app-grid" />
      
      <div className="flex-col items-center text-center px-4 relative z-10 pt-32 pb-20" style={{ maxWidth: '1100px', margin: '0 auto', minHeight: '100%', gap: '2.5rem' }}>
        
        <div className="flex-col items-center gap-6">
          <div className="badge badge-violet animate-in" style={{ padding: '0.4rem 1rem', fontSize: '0.8rem' }}>
            <Activity size={14} className="mr-2" />
            v2.0 Fraud Intelligence Engine Active
          </div>
          
          <h1 className="animate-in delay-1" style={{ fontSize: 'clamp(2.5rem, 8vw, 4.5rem)', lineHeight: '1.1', margin: 0 }}>
            Autonomous <br/>
            <span className="gradient-text-violet">Fraud Detection Pipeline</span>
          </h1>
          
          <p className="text-secondary animate-in delay-2" style={{ fontSize: 'clamp(1rem, 2vw, 1.25rem)', maxWidth: '800px', margin: '0 auto' }}>
            A production-grade MLOps ecosystem for high-precision fraud analytics. 
            Monitor data drift, automate training, and secure your transaction lifecycle with advanced AI.
          </p>
          
          <div className="flex-row gap-4 animate-in delay-3 mt-4">
            <button className="btn btn-primary" style={{ padding: '0.8rem 2.5rem', fontSize: '1.1rem' }} onClick={() => navigate('/dashboard')}>
              <Play size={18} /> Open Dashboard
            </button>
          </div>
        </div>

        <div className="grid-3 animate-in delay-4 w-full mt-12" style={{ gap: '1.5rem', alignItems: 'stretch' }}>
          <div className="glass-card p-8 flex-col items-start text-left gap-6">
            <div style={{ background: 'var(--violet-dim)', color: 'var(--violet)', padding: '0.8rem', borderRadius: '12px' }}>
              <Activity size={24} />
            </div>
            <div className="flex-col gap-2">
              <h3 style={{ fontSize: '1.25rem', fontWeight: 700 }}>Drift Monitoring</h3>
              <p className="text-secondary text-sm" style={{ lineHeight: '1.6' }}>
                Continuous data validation and concept drift detection using Evidently AI.
              </p>
            </div>
          </div>
          <div className="glass-card p-8 flex-col items-start text-left gap-6">
            <div style={{ background: 'var(--cyan-dim)', color: 'var(--cyan)', padding: '0.8rem', borderRadius: '12px' }}>
              <GitMerge size={24} />
            </div>
            <div className="flex-col gap-2">
              <h3 style={{ fontSize: '1.25rem', fontWeight: 700 }}>MLOps Automation</h3>
              <p className="text-secondary text-sm" style={{ lineHeight: '1.6' }}>
                Self-healing ML workflows and automated model retraining via Prefect.
              </p>
            </div>
          </div>
          <div className="glass-card p-8 flex-col items-start text-left gap-6">
            <div style={{ background: 'var(--emerald-dim)', color: 'var(--emerald)', padding: '0.8rem', borderRadius: '12px' }}>
              <Server size={24} />
            </div>
            <div className="flex-col gap-2">
              <h3 style={{ fontSize: '1.25rem', fontWeight: 700 }}>Precision Analytics</h3>
              <p className="text-secondary text-sm" style={{ lineHeight: '1.6' }}>
                Ensemble learning models optimized for imbalanced fraud datasets.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Landing;
