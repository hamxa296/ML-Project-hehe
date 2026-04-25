import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import TransactionsList from './pages/TransactionsList';
import InvestigationView from './pages/InvestigationView';
import Landing from './pages/Landing';
import SimulationEngine from './pages/SimulationEngine';
import PatternExplorer from './pages/PatternExplorer';
import './index.css';

function App() {
  return (
    <Router>
      <div className="flex-row" style={{ width: '100vw', height: '100vh', background: 'var(--bg-space)', position: 'relative' }}>
        <div className="app-bg" />
        <div className="app-grid" />
        
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/*" element={
            <div className="flex-row" style={{ width: '100%', height: '100%' }}>
              <Sidebar />
              <main style={{ flex: 1, position: 'relative' }}>
                <div className="scroll-container">
                  <Routes>
                    <Route path="/dashboard" element={<Dashboard />} />
                    <Route path="/simulation" element={<SimulationEngine />} />
                    <Route path="/patterns" element={<PatternExplorer />} />
                    <Route path="/transactions" element={<TransactionsList />} />
                    <Route path="/tx/:id" element={<InvestigationView />} />
                  </Routes>
                </div>
              </main>
            </div>
          } />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
