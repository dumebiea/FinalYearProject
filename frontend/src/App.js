import React, { useState, useEffect } from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
  Outlet,
} from 'react-router-dom';
import { getMe } from './api';
import { ThemeProvider } from './contexts/ThemeContext';
import Landing   from './pages/Landing';
import Login     from './pages/Login';
import Dashboard from './pages/Dashboard';
import Classify  from './pages/Classify';
import Alerts    from './pages/Alerts';
import Upload    from './pages/Upload';
import Navbar         from './components/Navbar';
import ProtectedRoute from './components/ProtectedRoute';
import HowToUse       from './pages/HowToUse';
import './styles/App.css';

/* Wraps every authenticated page with the sidebar + main-content shell */
function AppLayout({ user, setUser }) {
  return (
    <div className="layout">
      {user && <Navbar user={user} setUser={setUser} />}
      <div className={`main-content${!user ? ' full-width' : ''}`}>
        <Outlet />
      </div>
    </div>
  );
}

function App() {
  const [user, setUser]       = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) { setLoading(false); return; }

    getMe()
      .then((response) => {
        const payload = response.data.user || response.data;
        setUser(payload);
      })
      .catch(() => {
        localStorage.removeItem('token');
        localStorage.removeItem('user');
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className="loading">Loading...</div>;
  }

  return (
    <ThemeProvider>
      <Router>
        <Routes>
          {/* ── Landing page — full screen, no sidebar ── */}
          <Route path="/" element={<Landing user={user} />} />

          {/* ── All other pages — wrapped in the sidebar layout ── */}
          <Route element={<AppLayout user={user} setUser={setUser} />}>
            <Route
              path="/login"
              element={user ? <Navigate to="/dashboard" /> : <Login setUser={setUser} />}
            />
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <Dashboard user={user} />
                </ProtectedRoute>
              }
            />
            <Route
              path="/classify"
              element={
                <ProtectedRoute>
                  <Classify />
                </ProtectedRoute>
              }
            />
            <Route
              path="/upload"
              element={
                <ProtectedRoute>
                  <Upload />
                </ProtectedRoute>
              }
            />
            <Route
              path="/alerts"
              element={
                <ProtectedRoute>
                  <Alerts user={user} />
                </ProtectedRoute>
              }
            />
            <Route
              path="/how-to-use"
              element={
                <ProtectedRoute>
                  <HowToUse />
                </ProtectedRoute>
              }
            />
          </Route>
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;
