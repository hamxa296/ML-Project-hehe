import { useState, useMemo } from 'react';
import { Search, ChevronRight, Hash, UploadCloud, Loader2, AlertCircle } from 'lucide-react';
import { batchPredict } from '../services/api';

const TransactionsList = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    try {
      const response = await batchPredict(file);
      const results = response.batch_results.map((res, index) => ({
        id: `TXN-${1000 + index}`,
        probability: res.probability,
        isFraud: res.is_fraud_pred,
        riskLevel: res.is_fraud_pred === 1 ? 'severe' : res.probability > 0.3 ? 'suspicious' : 'safe'
      }));
      setTransactions(results);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const filteredTxns = useMemo(() => {
    return transactions.filter(t => {
      const search = searchTerm.toLowerCase();
      if (search.includes('fraud') || search.includes('severe')) return t.riskLevel === 'severe';
      if (search.includes('suspicious')) return t.riskLevel === 'suspicious';
      if (search.includes('safe')) return t.riskLevel === 'safe';
      return t.id.toLowerCase().includes(search);
    });
  }, [searchTerm, transactions]);

  return (
    <div className="scroll-container flex-col gap-6 animate-slide-up">
      <header className="flex-row justify-between items-end stagger-1" style={{ marginBottom: '1rem' }}>
        <div className="flex-col">
          <span className="section-title"><Hash size={16}/> Batch Inference</span>
          <h1 className="text-gradient-animate">Transaction Predictions</h1>
        </div>
        
        <div className="flex-row gap-4 items-center">
          <div style={{ position: 'relative' }}>
            <input 
              type="file" 
              accept=".csv"
              onChange={handleFileUpload}
              style={{ display: 'none' }} 
              id="file-upload" 
            />
            <label 
              htmlFor="file-upload" 
              style={{
                background: 'var(--violet)', color: 'white', padding: '0.75rem 1.5rem', borderRadius: '8px', 
                cursor: loading ? 'not-allowed' : 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem',
                opacity: loading ? 0.7 : 1
              }}
            >
              {loading ? <Loader2 size={18} className="animate-spin" /> : <UploadCloud size={18} />}
              Upload CSV
            </label>
          </div>

          <div style={{ flex: 1, minWidth: '250px', position: 'relative' }}>
            <Search size={18} style={{ position: 'absolute', top: '14px', left: '20px', color: 'var(--accent)' }} />
            <input 
              type="text" 
              placeholder='Filter predictions...' 
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
        </div>
      </header>

      {error && (
        <div className="p-4 rounded-md flex items-center gap-2" style={{ background: 'rgba(244, 63, 94, 0.1)', color: 'var(--rose)', border: '1px solid var(--rose)' }}>
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      <div className="glass-card stagger-2" style={{ border: '1px solid var(--border-glass)', overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ background: 'rgba(255,255,255,0.04)', borderBottom: '1px solid var(--border-glass)' }}>
              <th className="p-6 text-secondary" style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Reference ID</th>
              <th className="p-6 text-secondary" style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Fraud Probability</th>
              <th className="p-6 text-secondary" style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Risk Matrix</th>
              <th className="p-6 text-secondary" style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}></th>
            </tr>
          </thead>
          <tbody>
            {filteredTxns.map((txn, index) => (
              <tr 
                key={index} 
                className={`animate-slide-up hover:bg-[rgba(255,255,255,0.05)] transition-all`}
                style={{ borderTop: '1px solid var(--border-glass)', cursor: 'pointer', animationDelay: `${index * 0.05}s` }}
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
                <td className="p-6" style={{ fontSize: '1.2rem', fontWeight: '800' }}>{(txn.probability * 100).toFixed(2)}%</td>
                <td className="p-6">
                  <span className="badge" style={{ 
                    background: txn.riskLevel === 'safe' ? 'var(--safe-glow)' : txn.riskLevel === 'severe' ? 'var(--severe-glow)' : 'var(--suspicious-glow)',
                    color: txn.riskLevel === 'safe' ? 'var(--safe)' : txn.riskLevel === 'severe' ? 'var(--severe)' : 'var(--suspicious)',
                    border: `1px solid ${txn.riskLevel === 'safe' ? 'var(--safe)' : txn.riskLevel === 'severe' ? 'var(--severe)' : 'var(--suspicious)'}`,
                    boxShadow: `0 0 10px ${txn.riskLevel === 'safe' ? 'var(--safe-glow)' : txn.riskLevel === 'severe' ? 'var(--severe-glow)' : 'var(--suspicious-glow)'}`
                  }}>
                    {txn.riskLevel.toUpperCase()}
                  </span>
                </td>
                <td className="p-6 text-right">
                    <ChevronRight size={20} className="chevron-icon text-muted" style={{ transition: 'all 0.3s' }} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {filteredTxns.length === 0 && (
          <div className="p-12 text-center text-secondary flex flex-col items-center gap-2" style={{ fontSize: '1.1rem' }}>
            {loading ? <Loader2 size={32} className="animate-spin text-violet" /> : "Upload a CSV file to see predictions."}
          </div>
        )}
      </div>
    </div>
  );
};

export default TransactionsList;
