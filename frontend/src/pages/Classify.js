import React, { useState } from 'react';
import { predictTweet } from '../api';
import '../styles/Classify.css';

const MODEL_OPTIONS = [
  { key: 'bert',     label: 'NaijaSenti',          sub: 'XLM-RoBERTa' },
  { key: 'svm',      label: 'SVM',                  sub: 'TF-IDF + SVM' },
  { key: 'lr',       label: 'Logistic Regression',  sub: 'TF-IDF + LR' },
  { key: 'ensemble', label: 'Ensemble',              sub: 'All Models Combined' },
];

function Classify() {
  const [tweet, setTweet] = useState('');
  const [selectedModel, setSelectedModel] = useState('bert');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleAnalyze = async () => {
    if (!tweet.trim()) {
      setError('Please enter a tweet to analyze');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const response = await predictTweet(tweet, selectedModel);
      const data = response.data;

      if (data.ensemble) {
        setResult({
          isEnsemble: true,
          individual: data.individual,
          label: data.label,
          confidence: data.confidence,
          votes: data.votes,
          total: data.total,
          original_tweet: data.tweet || tweet,
          processed_text: data.cleaned || '',
          message: data.message || '',
        });
      } else {
        setResult({
          isEnsemble: false,
          sentiment: data.label || 'NEUTRAL',
          confidence: data.confidence || 0,
          original_tweet: data.tweet || tweet,
          processed_text: data.cleaned || '',
          model: data.model_used || 'BERT',
          message: data.message || '',
        });
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to analyze tweet. Please try again.');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (sentiment) => {
    if (sentiment === 'Negative' || sentiment === 'NEGATIVE') return '#C0392B';
    if (sentiment === 'Positive' || sentiment === 'POSITIVE') return '#27AE60';
    return '#888888';
  };

  const getSentimentEmoji = (sentiment) => {
    if (sentiment === 'Negative' || sentiment === 'NEGATIVE') return '🔴';
    if (sentiment === 'Positive' || sentiment === 'POSITIVE') return '🟢';
    return '⚪';
  };

  const getInterpretation = (sentiment) => {
    if (sentiment === 'Negative' || sentiment === 'NEGATIVE')
      return 'This tweet shows signs of food-related distress. Negative sentiment about food prices, hunger, or scarcity has been detected.';
    if (sentiment === 'Positive' || sentiment === 'POSITIVE')
      return 'This tweet shows a positive food security signal. Expressions of relief, food access, or welfare support have been detected.';
    return 'This tweet does not show strong positive or negative sentiment signals about food security.';
  };

  return (
    <div className="classify-container">
      <div className="classify-header">
        <h1>Classify a Tweet</h1>
        <p className="classify-subtitle">
          Type or paste any tweet to analyse its food crisis sentiment
        </p>
      </div>

      {/* Input Section */}
      <div className="card classify-card">
        <textarea
          className="tweet-input"
          placeholder="Type a tweet here e.g. Rice don reach 120k per bag for Lagos..."
          value={tweet}
          onChange={(e) => setTweet(e.target.value)}
          rows="5"
        />
        <div className="char-count">{tweet.length} characters</div>

        {/* Model Selector */}
        <div className="model-selector">
          <span className="model-selector-label">Choose model:</span>
          <div className="model-btn-group">
            {MODEL_OPTIONS.map((opt) => (
              <button
                key={opt.key}
                className={`model-btn ${selectedModel === opt.key ? 'active' : ''}`}
                onClick={() => setSelectedModel(opt.key)}
              >
                <span className="model-btn-name">{opt.label}</span>
                <span className="model-btn-sub">{opt.sub}</span>
              </button>
            ))}
          </div>
        </div>

        {error && <div className="error-message">{error}</div>}

        <button
          className="analyze-btn"
          onClick={handleAnalyze}
          disabled={loading || !tweet.trim()}
        >
          {loading ? (
            <><span className="spinner"></span> Analyzing...</>
          ) : (
            'Analyse Tweet'
          )}
        </button>
      </div>

      {/* Single Model Results */}
      {result && !result.isEnsemble && (
        <div className="card results-card">
          <div
            className="sentiment-badge"
            style={{ backgroundColor: getSentimentColor(result.sentiment) }}
          >
            <div className="badge-emoji">{getSentimentEmoji(result.sentiment)}</div>
            <div className="badge-text">{result.sentiment}</div>
          </div>

          <div className="confidence-display">
            Confidence: {parseFloat(result.confidence).toFixed(1)}%
          </div>

          <div className="results-divider"></div>

          <div className="results-columns">
            <div className="result-column">
              <h4>Original Tweet</h4>
              <p className="result-text">{result.original_tweet}</p>
            </div>
            <div className="result-column">
              <h4>Processed Text</h4>
              <p className="result-text">{result.processed_text}</p>
            </div>
          </div>

          <div className="model-info">Model: {result.model}</div>

          <div className="interpretation-box">
            {getInterpretation(result.sentiment)}
          </div>
        </div>
      )}

      {/* Ensemble Results */}
      {result && result.isEnsemble && (
        <div className="card results-card">
          <div className="ensemble-title">Ensemble Analysis</div>

          {/* Individual model cards */}
          <div className="ensemble-individual">
            {result.individual.map((item, idx) => (
              <div key={idx} className="ensemble-model-card">
                <div className="ensemble-model-name">{item.model}</div>
                <div
                  className="ensemble-model-badge"
                  style={{ backgroundColor: getSentimentColor(item.label) }}
                >
                  {getSentimentEmoji(item.label)} {item.label}
                </div>
                <div className="ensemble-model-conf">
                  {parseFloat(item.confidence).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>

          {/* Final verdict */}
          <div className="ensemble-verdict">
            <div
              className="sentiment-badge"
              style={{ backgroundColor: getSentimentColor(result.label) }}
            >
              <div className="badge-emoji">{getSentimentEmoji(result.label)}</div>
              <div className="badge-text">{result.label}</div>
            </div>

            <div className="ensemble-vote-line">
              {result.votes} out of {result.total} models agreed: <strong>{result.label}</strong>
            </div>
            <div className="confidence-display">
              Ensemble confidence: {parseFloat(result.confidence).toFixed(1)}%
            </div>
          </div>

          <div className="results-divider"></div>

          <div className="results-columns">
            <div className="result-column">
              <h4>Original Tweet</h4>
              <p className="result-text">{result.original_tweet}</p>
            </div>
            <div className="result-column">
              <h4>Processed Text</h4>
              <p className="result-text">{result.processed_text}</p>
            </div>
          </div>

          <div className="interpretation-box">
            {getInterpretation(result.label)}
          </div>
        </div>
      )}

      {result && (
        <p className="disclaimer-text">
          Note: Results are based on a model trained on Lagos food crisis discourse. Accuracy may vary for non-food related tweets.
        </p>
      )}
    </div>
  );
}

export default Classify;
