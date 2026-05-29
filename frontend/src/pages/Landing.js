import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useTheme } from '../contexts/ThemeContext';
import '../styles/Landing.css';

const FEATURES = [
  {
    icon: '📡',
    title: 'Real-Time Monitoring',
    desc: 'Continuously tracks Twitter discussions across all Lagos Local Government Areas for food-related signals.',
  },
  {
    icon: '🤖',
    title: 'AI Classification',
    desc: 'Powered by NaijaSenti — a BERT model fine-tuned specifically for Nigerian English sentiment analysis.',
  },
  {
    icon: '🚨',
    title: 'Crisis Forecasting',
    desc: '7-day sentiment predictions with automatic crisis threshold alerts to enable early intervention.',
  },
  {
    icon: '📊',
    title: 'Regional Analysis',
    desc: 'Sentiment heatmaps, trend charts, and crisis risk rankings broken down by district.',
  },
];

function Landing({ user }) {
  const navigate = useNavigate();
  const { isDark, toggleTheme } = useTheme();

  const handleEnter = () => navigate(user ? '/dashboard' : '/login');

  return (
    <div className="landing-page">

      {/* ── Top bar ── */}
      <div className="landing-topbar">
        <div className="landing-topbar-brand">
          <span className="landing-topbar-dot" />
          Food Crisis Monitor
        </div>
        <button
          className="landing-theme-btn"
          onClick={toggleTheme}
          aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
          title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {isDark ? '☀️' : '🌙'}
        </button>
      </div>

      {/* ── Hero ── */}
      <main className="landing-hero">
        <div className="landing-hero-inner">
          <span className="landing-badge">
            AI-Powered &nbsp;·&nbsp; Food Security &nbsp;·&nbsp; Lagos, Nigeria
          </span>

          <h1 className="landing-title">
            Early Detection of Food Crisis<br />
            Signals in Lagos Using NLP<br />
            and Sentiment Analysis
          </h1>

          <p className="landing-desc">
            This system monitors real-time Twitter discussions in Lagos to detect early signs
            of food crisis using Artificial Intelligence and sentiment analysis. It automatically
            classifies public sentiment, identifies at-risk regions, and generates 7-day forecasts
            to help policymakers respond before situations escalate.
          </p>

          <div className="landing-cta-row">
            <button className="landing-cta" onClick={handleEnter}>
              {user ? 'Enter Dashboard' : 'Get Started'} &rarr;
            </button>
            {user && (
              <p className="landing-logged-in">
                Signed in as <strong>{user.username || user.email}</strong>
              </p>
            )}
          </div>
        </div>
      </main>

      {/* ── Feature cards ── */}
      <section className="landing-features-section">
        <h2 className="landing-features-heading">What This System Does</h2>
        <div className="landing-features">
          {FEATURES.map((f, i) => (
            <div key={i} className="landing-feature-card">
              <div className="landing-feature-icon">{f.icon}</div>
              <h3 className="landing-feature-title">{f.title}</h3>
              <p className="landing-feature-desc">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="landing-footer">
        <div className="landing-footer-divider" />
        <div className="landing-footer-content">
          <div className="landing-footer-uni">
            <strong>Covenant University</strong> &mdash; Department of Computer Science and Information Systems
          </div>
          <div className="landing-footer-dev">
            Developed by <strong>Emeka-Anyaeji Dumebi</strong> &nbsp;&bull;&nbsp; Final Year Project &nbsp;&bull;&nbsp; 2026
          </div>
          <div className="landing-footer-dev">
            Supervised by <strong>MR Damilola O. Osofuye</strong> 
          </div>
        </div>
      </footer>

    </div>
  );
}

export default Landing;
