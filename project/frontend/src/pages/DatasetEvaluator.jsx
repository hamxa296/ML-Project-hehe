import { useState } from 'react';
import { UploadCloud, Loader2, AlertCircle, ChevronRight, Activity } from 'lucide-react';
import { batchPredict } from '../services/api';

const DatasetEvaluator = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [filterMode, setFilterMode] = useState('all'); // 'all', 'fraud', 'safe'
  const [displayLimit, setDisplayLimit] = useState(50);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    setResults(null);
    setDisplayLimit(50);
    try {
      const response = await batchPredict(file);
      setResults(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="scroll-container flex-col gap-6 animate-slide-up">
      <header className="flex-row justify-between items-end stagger-1" style={{ marginBottom: '1rem' }}>
        <div className="flex-col">
          <span className="section-title"><Activity size={16}/> Model Testing</span>
          <h1 className="text-gradient-animate">Dataset Evaluator</h1>
        </div>
        
        <div className="flex-row gap-4 items-center">
          <div style={{ position: 'relative' }}>
            <input 
              type="file" 
              accept=".csv"
              onChange={handleFileUpload}
              style={{ display: 'none' }} 
              id="dataset-upload" 
            />
            <label 
              htmlFor="dataset-upload" 
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
        </div>
      </header>

      {error && (
        <div className="p-4 rounded-md flex items-center gap-2" style={{ background: 'rgba(244, 63, 94, 0.1)', color: 'var(--rose)', border: '1px solid var(--rose)' }}>
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      {results && results.metrics && (
        <div className="stagger-2 mb-4" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1rem' }}>
          {[
            { label: 'Accuracy', val: (results.metrics.accuracy * 100).toFixed(2) + '%' },
            { label: 'Precision', val: (results.metrics.precision * 100).toFixed(2) + '%' },
            { label: 'Recall', val: (results.metrics.recall * 100).toFixed(2) + '%' },
            { label: 'F1-Score', val: (results.metrics.f1 * 100).toFixed(2) + '%' },
            { label: 'ROC-AUC', val: (results.metrics.roc_auc * 100).toFixed(2) + '%' }
          ].map((m, i) => (
             <div key={i} className="glass-card p-4 flex-col gap-2 text-center" style={{ border: '1px solid var(--border-glass)' }}>
                <span className="text-secondary text-sm">{m.label}</span>
                <span className="font-bold text-lg text-emerald">{m.val}</span>
             </div>
          ))}
        </div>
      )}

      {results && results.results_table && (
        <div className="flex-col gap-4 stagger-3">
          <div className="flex-row gap-2">
            {(results.metrics ? ['all', 'fraud', 'safe', 'false-positives', 'false-negatives'] : ['all', 'fraud', 'safe']).map(mode => (
              <button 
                key={mode}
                onClick={() => setFilterMode(mode)}
                className={`btn ${filterMode === mode ? 'btn-primary' : 'btn-outline'}`}
                style={{ 
                  textTransform: 'capitalize', 
                  borderColor: filterMode === mode ? (mode === 'fraud' ? 'var(--rose)' : mode === 'safe' ? 'var(--emerald)' : mode.includes('false') ? 'var(--amber)' : 'var(--border-bright)') : 'var(--border-default)',
                  color: filterMode === mode ? (mode === 'fraud' ? 'var(--rose)' : mode === 'safe' ? 'var(--emerald)' : mode.includes('false') ? 'var(--amber)' : 'var(--text-primary)') : 'var(--text-secondary)'
                }}
              >
                {mode.replace('-', ' ')}
              </button>
            ))}
          </div>
          
          {(() => {
            const filteredRows = results.results_table.filter(row => {
              if (filterMode === 'fraud') return row.prediction === 1;
              if (filterMode === 'safe') return row.prediction === 0;
              if (filterMode === 'false-positives') return row.prediction === 1 && row.true_label === 0;
              if (filterMode === 'false-negatives') return row.prediction === 0 && row.true_label === 1;
              return true;
            });
            
            return (
          <div className="glass-card" style={{ border: '1px solid var(--border-glass)', overflow: 'hidden' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ background: 'rgba(255,255,255,0.04)', borderBottom: '1px solid var(--border-glass)' }}>
                <th className="p-6 text-secondary" style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Row Index</th>
                <th className="p-6 text-secondary" style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Fraud Probability</th>
                <th className="p-6 text-secondary" style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Prediction</th>
                {results.metrics && <th className="p-6 text-secondary" style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Actual / Status</th>}
                <th className="p-6 text-secondary" style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}></th>
              </tr>
            </thead>
            <tbody>
              {filteredRows.slice(0, displayLimit).map((row) => (
                <tr 
                  key={row.index} 
                  className="hover:bg-[rgba(255,255,255,0.05)] transition-all"
                  style={{ borderTop: '1px solid var(--border-glass)' }}
                >
                  <td className="p-6" style={{ fontFamily: "'Space Grotesk', monospace", color: 'var(--accent)', fontWeight: 600 }}>#{row.index}</td>
                  <td className="p-6" style={{ fontSize: '1.1rem', fontWeight: '800' }}>{(row.probability * 100).toFixed(2)}%</td>
                  <td className="p-6">
                    <span className="badge" style={{ 
                      background: row.prediction === 1 ? 'var(--rose-glow)' : 'var(--safe-glow)',
                      color: row.prediction === 1 ? 'var(--rose)' : 'var(--safe)',
                      border: `1px solid ${row.prediction === 1 ? 'var(--rose)' : 'var(--safe)'}`
                    }}>
                      {row.prediction === 1 ? 'FRAUD' : 'SAFE'}
                    </span>
                  </td>
                  {results.metrics && (
                    <td className="p-6">
                      {row.prediction !== row.true_label ? (
                        <span className="badge" style={{ background: 'var(--amber-glow)', color: 'var(--amber)', border: '1px solid var(--amber)' }}>
                          {row.prediction === 1 ? 'FALSE POSITIVE' : 'FALSE NEGATIVE'}
                        </span>
                      ) : (
                        <span className="text-secondary text-sm">Correct Match</span>
                      )}
                    </td>
                  )}
                  <td className="p-6 text-right">
                      <ChevronRight size={20} className="text-muted" />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          
          {filteredRows.length > displayLimit ? (
            <div className="p-4 flex-col items-center justify-center border-t" style={{ borderColor: 'var(--border-glass)', background: 'rgba(255,255,255,0.01)' }}>
               <span className="text-secondary text-sm mb-3">Showing {displayLimit} of {filteredRows.length} matches</span>
               <button 
                 onClick={() => setDisplayLimit(prev => prev + 50)}
                 className="btn btn-outline"
               >
                 Load 50 More
               </button>
            </div>
          ) : (
            <div className="p-4 text-center text-secondary text-sm border-t" style={{ borderColor: 'var(--border-glass)' }}>
               Showing all {filteredRows.length} matches.
            </div>
          )}
          </div>
          );
          })()}
        </div>
      )}

      {!results && !loading && !error && (
        <div className="p-12 glass-card text-center text-secondary flex flex-col items-center gap-2 stagger-2">
           Upload a dataset CSV to begin evaluation. If the CSV contains an &apos;isFraud&apos; column, evaluation metrics will be automatically computed.
        </div>
      )}
    </div>
  );
};

export default DatasetEvaluator;
