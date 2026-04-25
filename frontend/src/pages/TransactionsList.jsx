import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, ChevronRight, Hash } from 'lucide-react';
import { mockTransactions } from '../data/mockData';

const TransactionsList = () => {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');

  const filteredTxns = useMemo(() => {
    return mockTransactions.filter(t => {
      const search = searchTerm.toLowerCase();
      if (search.includes('high value')) return t.amount > 1000;
      if (search.includes('suspicious')) return t.riskLevel !== 'safe';
      return t.id.toLowerCase().includes(search) || t.merchant.toLowerCase().includes(search);
    });
  }, [searchTerm]);

  return (
    <div className="flex-col gap-6 animate-slide-up" style={{ height: '100%' }}>
      <header className="flex-row justify-between items-end stagger-1" style={{ marginBottom: '1rem' }}>
        <div className="flex-col">
          <span className="section-title"><Hash size={16}/> Database</span>
          <h1 className="text-gradient-animate">Transaction Feed</h1>
        </div>
        
        <div style={{ flex: 1, maxWidth: '400px', position: 'relative' }}>
          <Search size={18} style={{ position: 'absolute', top: '14px', left: '20px', color: 'var(--accent)' }} />
          <input 
            type="text" 
            placeholder='Try: "high value fraud"' 
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{ 
              width: '100%', padding: '0.85rem 1rem 0.85rem 3.5rem', 
              borderRadius: '16px', background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-glass)',
              color: 'white', fontSize: '0.95rem', outline: 'none', transition: 'all 0.3s',
              boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.2)'
            }}
            onFocus={(e) => { e.target.style.borderColor = 'var(--accent)'; e.target.style.boxShadow = '0 0 15px var(--accent-glow)'; }}
            onBlur={(e) => { e.target.style.borderColor = 'var(--border-glass)'; e.target.style.boxShadow = 'inset 0 2px 4px rgba(0,0,0,0.2)'; }}
          />
        </div>
      </header>

      <div className="glass-card stagger-2" style={{ border: '1px solid var(--border-glass)', overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ background: 'rgba(255,255,255,0.04)', borderBottom: '1px solid var(--border-glass)' }}>
              <th className="p-6 text-secondary" style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Reference ID</th>
              <th className="p-6 text-secondary" style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Merchant Entity</th>
              <th className="p-6 text-secondary" style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Volume</th>
              <th className="p-6 text-secondary" style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Time Window</th>
              <th className="p-6 text-secondary" style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Risk Matrix</th>
              <th className="p-6 text-secondary" style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}></th>
            </tr>
          </thead>
          <tbody>
            {filteredTxns.map((txn, index) => (
              <tr 
                key={txn.id} 
                className={`animate-slide-up hover:bg-[rgba(255,255,255,0.05)] transition-all`}
                style={{ borderTop: '1px solid var(--border-glass)', cursor: 'pointer', animationDelay: `${index * 0.05}s` }}
                onClick={() => navigate(`/tx/${txn.id}`)}
                onMouseEnter={(e) => {
                   const glowColor = txn.riskLevel === 'safe' ? 'var(--safe-glow)' : txn.riskLevel === 'severe' ? 'var(--severe-glow)' : 'var(--suspicious-glow)';
                   e.currentTarget.style.boxShadow = `inset 4px 0 0 ${txn.riskLevel === 'safe' ? 'var(--safe)' : txn.riskLevel === 'severe' ? 'var(--severe)' : 'var(--suspicious)'}, inset 0 0 20px ${glowColor}`;
                   e.currentTarget.querySelector('.chevron-icon').style.transform = 'translateX(5px)';
                   e.currentTarget.querySelector('.chevron-icon').style.color = 'var(--text-primary)';
                }}
                onMouseLeave={(e) => {
                   e.currentTarget.style.boxShadow = 'none';
                   e.currentTarget.querySelector('.chevron-icon').style.transform = 'translateX(0)';
                   e.currentTarget.querySelector('.chevron-icon').style.color = 'var(--text-muted)';
                }}
              >
                <td className="p-6" style={{ fontFamily: "'Space Grotesk', monospace", color: 'var(--accent)', fontWeight: 600 }}>{txn.id}</td>
                <td className="p-6" style={{ fontWeight: '600', fontSize: '1.05rem' }}>{txn.merchant}</td>
                <td className="p-6" style={{ fontSize: '1.2rem', fontWeight: '800' }}>${txn.amount.toFixed(2)}</td>
                <td className="p-6 text-muted" style={{ fontSize: '0.9rem' }}>{txn.time}</td>
                <td className="p-6">
                  <span className="badge" style={{ 
                    background: txn.riskLevel === 'safe' ? 'var(--safe-glow)' : txn.riskLevel === 'severe' ? 'var(--severe-glow)' : 'var(--suspicious-glow)',
                    color: txn.riskLevel === 'safe' ? 'var(--safe)' : txn.riskLevel === 'severe' ? 'var(--severe)' : 'var(--suspicious)',
                    border: `1px solid ${txn.riskLevel === 'safe' ? 'var(--safe)' : txn.riskLevel === 'severe' ? 'var(--severe)' : 'var(--suspicious)'}`,
                    boxShadow: `0 0 10px ${txn.riskLevel === 'safe' ? 'var(--safe-glow)' : txn.riskLevel === 'severe' ? 'var(--severe-glow)' : 'var(--suspicious-glow)'}`
                  }}>
                    {txn.riskLevel}
                  </span>
                </td>
                <td className="p-6 text-right">
                    <ChevronRight size={20} className="chevron-icon text-muted" style={{ transition: 'all 0.3s' }} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {filteredTxns.length === 0 && <div className="p-12 text-center text-secondary" style={{ fontSize: '1.1rem' }}>No matching security logs found.</div>}
      </div>
    </div>
  );
};

export default TransactionsList;
