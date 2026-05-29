import React, { useState, useEffect, useRef } from 'react';
import { getStreamStatus, getSystemHealth } from '../api';
import ModelSelector from './ModelSelector';
import '../styles/LiveStream.css';

const MODEL_SHORT = {
  'NaijaSenti (XLM-RoBERTa)': 'NaijaSenti',
  'SVM': 'SVM',
  'Logistic Regression': 'LR',
};

function LiveStream() {
  const [tweets, setTweets]               = useState([]);
  const [status, setStatus]               = useState('loading');
  const [totalStreamed, setTotalStreamed]  = useState(0);
  const [model, setModel]                 = useState('bert');
  const [activeSystemModel, setActiveSystemModel] = useState('bert');
  const [isPaused, setIsPaused]           = useState(false);
  const [showConfirm, setShowConfirm]     = useState(false);
  const [resetMessage, setResetMessage]   = useState('');
  const [resetKey, setResetKey]           = useState(0);

  // Refs keep values fresh inside the interval closure
  const activeModelRef  = useRef('bert');
  const isPausedRef     = useRef(false);
  // Records the DB total on first poll of each session; display = rawTotal - base
  const sessionBaseRef  = useRef(null);

  // Fetch which model is currently active on the backend
  useEffect(() => {
    getSystemHealth()
      .then(res => {
        const m = res.data.model === 'svm' ? 'svm' : 'bert';
        setActiveSystemModel(m);
        activeModelRef.current = m;
        setModel(m);
      })
      .catch(() => {});
  }, []);

  // Poll stream-status; resetKey forces a fresh interval on reset
  useEffect(() => {
    const fetchStream = async () => {
      if (isPausedRef.current) return; // silently skip while paused
      try {
        const modelParam = model !== activeModelRef.current ? model : null;
        const response = await getStreamStatus(modelParam);
        setTweets(response.data.recent_tweets || []);
        setStatus(response.data.status || 'streaming');
        const rawTotal = response.data.total_streamed || 0;
        if (sessionBaseRef.current === null) {
          sessionBaseRef.current = rawTotal; // anchor to DB count at session start
        }
        setTotalStreamed(rawTotal - sessionBaseRef.current);
      } catch {
        setStatus('error');
      }
    };

    fetchStream();
    const interval = setInterval(fetchStream, 10000);
    return () => clearInterval(interval);
  }, [model, activeSystemModel, resetKey]);

  const handlePauseResume = () => {
    const next = !isPaused;
    isPausedRef.current = next;
    setIsPaused(next);
  };

  const handleResetClick = () => setShowConfirm(true);

  const handleConfirmReset = () => {
    setShowConfirm(false);
    isPausedRef.current = false;
    setIsPaused(false);
    setTweets([]);
    setTotalStreamed(0);
    setStatus('streaming');
    // Reset base so next poll re-anchors to current DB total → display starts at 0
    sessionBaseRef.current = null;
    // Bump resetKey → re-runs the polling effect immediately
    setResetKey(k => k + 1);
    setResetMessage('Stream display reset successfully. All data has been preserved.');
    setTimeout(() => setResetMessage(''), 4000);
  };

  const handleCancelReset = () => setShowConfirm(false);

  const getSentimentColor = (label) => {
    if (label === 'Negative') return '#C0392B';
    if (label === 'Positive') return '#27AE60';
    return '#888888';
  };

  return (
    <div className="live-stream-container">
      {/* Header row */}
      <div className="stream-header">
        <h3>📡 Live Data Stream</h3>

        <div className="stream-right">
          {/* Status indicator */}
          <div className="stream-status">
            <span className={`status-indicator ${isPaused ? 'paused-dot' : status}`}></span>
            <span className={`stream-status-label ${isPaused ? 'paused' : 'streaming'}`}>
              {isPaused ? 'Paused' : 'Streaming'}
            </span>
            <span className="tweet-count">({totalStreamed} tweets processed)</span>
          </div>

          {/* Controls */}
          <div className="stream-controls">
            <button
              className="stream-btn pause-btn"
              onClick={handlePauseResume}
              title={isPaused ? 'Resume stream' : 'Pause stream'}
            >
              {isPaused ? '▶' : '⏸'} {isPaused ? 'Resume' : 'Pause'}
            </button>
            <button
              className="stream-btn reset-btn"
              onClick={handleResetClick}
              title="Reset stream"
            >
              ↺ Reset
            </button>
          </div>
        </div>
      </div>

      {/* Reset success message */}
      {resetMessage && (
        <div className="stream-reset-msg">
          {resetMessage}
        </div>
      )}

      {/* Confirmation dialog */}
      {showConfirm && (
        <div className="stream-confirm-overlay">
          <div className="stream-confirm-box">
            <p className="stream-confirm-text">
              Reset stream display and counter? Your data will not be deleted.
            </p>
            <div className="stream-confirm-actions">
              <button className="stream-confirm-yes" onClick={handleConfirmReset}>
                Confirm Reset
              </button>
              <button className="stream-confirm-no" onClick={handleCancelReset}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      <ModelSelector
        value={model}
        onChange={setModel}
        label="Classify live feed with:"
      />

      <div className="tweets-feed">
        {tweets.length === 0 ? (
          <p className="empty-state">
            {isPaused ? 'Stream paused. Press Resume to continue.' : 'Waiting for stream data...'}
          </p>
        ) : (
          tweets.map((tweet, idx) => (
            <div key={idx} className={`tweet-item${tweet.ensemble ? ' ensemble' : ''}`}>
              <div className="tweet-text">{tweet.text}</div>

              {tweet.ensemble ? (
                <div className="tweet-meta ensemble-meta">
                  <div className="ensemble-mini-row">
                    {tweet.individual.map((ind, i) => (
                      <span key={i} className="ensemble-mini-item">
                        <span className="ensemble-model-short">
                          {MODEL_SHORT[ind.model] || ind.model}:
                        </span>
                        <span
                          className="ensemble-mini-badge"
                          style={{ backgroundColor: getSentimentColor(ind.label) }}
                        >
                          {ind.label}
                        </span>
                        <span className="ensemble-mini-conf">
                          {Number(ind.confidence).toFixed(0)}%
                        </span>
                      </span>
                    ))}
                    <span className="ensemble-arrow">→</span>
                    <span
                      className="sentiment-badge"
                      style={{ backgroundColor: getSentimentColor(tweet.label) }}
                    >
                      {tweet.label}
                    </span>
                    <span className="ensemble-votes">
                      ({tweet.votes}/{tweet.total})
                    </span>
                  </div>
                  <div className="ensemble-bottom-row">
                    <span className="confidence">
                      {Number(tweet.confidence).toFixed(1)}%
                    </span>
                    <span className="location">{tweet.location}</span>
                    <span className="time">
                      {new Date(tweet.created_at).toLocaleTimeString()}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="tweet-meta">
                  <span
                    className="sentiment-badge"
                    style={{ backgroundColor: getSentimentColor(tweet.label) }}
                  >
                    {tweet.label}
                  </span>
                  <span className="confidence">
                    {Number(tweet.confidence).toFixed(1)}%
                  </span>
                  <span className="location">{tweet.location}</span>
                  <span className="time">
                    {new Date(tweet.created_at).toLocaleTimeString()}
                  </span>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default LiveStream;
