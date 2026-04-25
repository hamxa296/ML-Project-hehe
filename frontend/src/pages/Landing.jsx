import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ShieldCheck, Cpu, Code2, Rocket, Globe, Zap, ArrowRight, BrainCircuit, Activity, Database } from 'lucide-react';

const Landing = () => {
  const navigate = useNavigate();
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <div className="scroll-container animate-slide-up" style={{ padding: 0, width: '100vw', overflowX: 'hidden', position: 'relative' }}>
      {/* Interactive Cursor Spotlight */}
      <div style={{
         position: 'fixed',
         top: 0, left: 0,
         width: '100vw', height: '100vh',
         pointerEvents: 'none',
         zIndex: 1,
         background: `radial-gradient(circle 500px at ${mousePosition.x}px ${mousePosition.y}px, rgba(197, 160, 89, 0.05), transparent 80%)`,
         transition: 'background 0.3s ease'
      }} />
      {/* Hero Section */}
      <section style={{ 
        minHeight: '90vh', 
        display: 'flex', 
        flexDirection: 'column', 
        justifyContent: 'center', 
        alignItems: 'center', 
        textAlign: 'center', 
        padding: '4rem 2rem',
        position: 'relative',
        width: '100%',
        zIndex: 2
      }}>
        <div className="badge stagger-1" style={{ background: 'var(--accent-glow)', color: 'var(--accent)', marginBottom: '2.5rem', border: '1px solid var(--accent)', padding: '8px 24px', filter: 'drop-shadow(0 0 10px rgba(59,130,246,0.5))' }}>
           v1.0.0 Neural Core Active
        </div>
        <h1 className="stagger-2 text-gradient-animate" style={{ fontSize: '5.5rem', lineHeight: '1', maxWidth: '1000px', margin: '0 auto 2rem auto', paddingBottom: '1rem' }}>
          Next-Gen Fraud <br/> 
          <span style={{ color: 'var(--accent)', WebkitTextFillColor: 'initial', filter: 'drop-shadow(0 0 15px rgba(59,130,246,0.3))' }}>Intelligence Defense</span>
        </h1>
        <p className="text-secondary stagger-3" style={{ fontSize: '1.5rem', maxWidth: '800px', margin: '0 auto 3.5rem auto', lineHeight: '1.6', fontWeight: 400 }}>
          Harnessing advanced machine learning to secure financial ecosystems. <br/> 
          Detect, simulate, and neutralize threats in real-time.
        </p>
        <div className="flex-row gap-4 stagger-4" style={{ justifyContent: 'center' }}>
          <button className="btn-action" style={{ padding: '1.4rem 3.5rem', fontSize: '1.2rem' }} onClick={() => navigate('/dashboard')}>
             Dashboard <ArrowRight size={22} style={{ marginLeft: '0.5rem' }} />
          </button>
        </div>
      </section>

      {/* Dashboard Feature Preview */}
      <section style={{ padding: '6rem 2rem', maxWidth: '1200px', margin: '0 auto', position: 'relative', zIndex: 2 }}>
... (existing feature preview section content) ...
      </section>

      {/* Tech Stack Sections */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3rem', padding: '6rem 2rem', maxWidth: '1200px', margin: '0 auto', position: 'relative', zIndex: 2 }}>
... (existing tech stack content) ...
      </div>

      {/* Minimal Footer */}
      <footer style={{ padding: '2rem', textAlign: 'center', borderTop: '1px solid var(--border-glass)', marginTop: '4rem', background: 'rgba(255,255,255,0.01)', position: 'relative', zIndex: 2 }}>
        <p className="text-muted" style={{ fontSize: '0.85rem', margin: 0, letterSpacing: '0.1em' }}>
           SYSTEM-01 // ORACLE AI SURVEILLANCE &copy; 2026
        </p>
      </footer>
    </div>
  );
};

export default Landing;
