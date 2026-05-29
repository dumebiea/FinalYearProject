import React, { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { useTheme } from '../contexts/ThemeContext';
import { isSoundEnabled, setSoundEnabled } from '../utils/alertSound';
import '../styles/Navbar.css';

function Navbar({ user, setUser }) {
  const [isOpen, setIsOpen] = useState(false);
  const [soundOn, setSoundOn] = useState(isSoundEnabled);
  const navigate = useNavigate();
  const { isDark, toggleTheme } = useTheme();

  const handleSoundToggle = () => {
    const next = !soundOn;
    setSoundOn(next);
    setSoundEnabled(next);
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
    navigate('/login');
  };

  const displayName = user?.username || user?.name || 'User';

  return (
    <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
      <div className="sidebar-header">
        <div className="brand">Food Crisis Monitor</div>
        <button className="hamburger" onClick={() => setIsOpen((prev) => !prev)}>
          ☰
        </button>
      </div>

      <nav className="sidebar-nav">
        <NavLink to="/dashboard" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          Dashboard
        </NavLink>
        <NavLink to="/classify" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          Classify Tweet
        </NavLink>
        <NavLink to="/upload" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          Upload Tweets
        </NavLink>
        <NavLink to="/alerts" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          Alerts
        </NavLink>
      </nav>

      <div className="sidebar-footer">
        <div className="sidebar-user">{displayName}</div>
        <div className="navbar-toggles-row">
          <button
            className="theme-toggle-btn theme-toggle-btn--half"
            onClick={toggleTheme}
            aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
            title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            <span className="theme-icon">{isDark ? '☀️' : '🌙'}</span>
            <span className="theme-label">{isDark ? 'Light' : 'Dark'}</span>
          </button>
          <button
            className={`sound-toggle-btn${soundOn ? '' : ' sound-toggle-btn--muted'}`}
            onClick={handleSoundToggle}
            aria-label={soundOn ? 'Mute alert sound' : 'Unmute alert sound'}
            title={soundOn ? 'Alert sound: On — click to mute' : 'Alert sound: Off — click to unmute'}
          >
            <span className="sound-icon">{soundOn ? '🔔' : '🔕'}</span>
          </button>
        </div>
        <NavLink
          to="/how-to-use"
          className={({ isActive }) => `how-to-use-btn${isActive ? ' how-to-use-btn--active' : ''}`}
        >
          <span className="how-to-use-icon">❓</span>
          <span className="how-to-use-label">How To Use</span>
        </NavLink>
        <button onClick={handleLogout} className="logout-btn">Logout</button>
      </div>
    </aside>
  );
}

export default Navbar;
