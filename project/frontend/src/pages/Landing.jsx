import { useNavigate } from 'react-router-dom';
import { Play, Database, Server, GitMerge, Activity, CheckCircle2 } from 'lucide-react';

const Landing = () => {
  const navigate = useNavigate();

  return (
    <div className="scroll-container p-0 w-full h-screen relative">
      <div className="app-bg" />
      <div className="app-grid" />
      
      <div className="flex-col items-center justify-center h-full text-center px-4 relative z-10" style={{ maxWidth: '1000px', margin: '0 auto' }}>
        
        <div className="badge badge-violet animate-in mb-8" style={{ padding: '0.4rem 1rem', fontSize: '0.8rem' }}>
          <Activity size={14} className="mr-2" />
          v2.0 ML Pipeline Engine Active
        </div>
        
        <h1 className="animate-in delay-1" style={{ fontSize: '4.5rem', marginBottom: '1.5rem' }}>
          Automate Your <br/>
          <span className="gradient-text-violet">Machine Learning Lifecycle</span>
        </h1>
        
        <p className="text-secondary animate-in delay-2" style={{ fontSize: '1.25rem', maxWidth: '700px', margin: '0 auto 3rem auto' }}>
          Continuous integration and deployment for AI models. Monitor data drift, track training metrics, and deploy to production with confidence.
        </p>
        
        <div className="flex-row gap-4 animate-in delay-3">
          <button className="btn btn-primary" style={{ padding: '0.8rem 2rem', fontSize: '1.1rem' }} onClick={() => navigate('/dashboard')}>
            <Play size={18} /> Open Dashboard
          </button>
          <button className="btn btn-ghost" style={{ padding: '0.8rem 2rem', fontSize: '1.1rem' }}>
            View Documentation
          </button>
        </div>

        <div className="grid-3 mt-16 animate-in delay-4 w-full">
          <div className="glass-card p-6 flex-col items-center text-center gap-4">
            <div style={{ background: 'var(--violet-dim)', color: 'var(--violet)', padding: '1rem', borderRadius: '50%' }}>
              <Database size={24} />
            </div>
            <h3>Automated EDA</h3>
            <p className="text-secondary text-sm">Real-time exploratory data analysis and feature drift monitoring.</p>
          </div>
          <div className="glass-card p-6 flex-col items-center text-center gap-4">
            <div style={{ background: 'var(--cyan-dim)', color: 'var(--cyan)', padding: '1rem', borderRadius: '50%' }}>
              <GitMerge size={24} />
            </div>
            <h3>CI/CD for Models</h3>
            <p className="text-secondary text-sm">Automated training pipelines, model versioning, and validation checks.</p>
          </div>
          <div className="glass-card p-6 flex-col items-center text-center gap-4">
            <div style={{ background: 'var(--emerald-dim)', color: 'var(--emerald)', padding: '1rem', borderRadius: '50%' }}>
              <Server size={24} />
            </div>
            <h3>1-Click Deploy</h3>
            <p className="text-secondary text-sm">Push approved models directly to high-performance inference endpoints.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Landing;
