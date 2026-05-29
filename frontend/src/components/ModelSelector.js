import React from 'react';
import '../styles/ModelSelector.css';

export const MODEL_OPTIONS = [
  { key: 'bert',     label: 'NaijaSenti',         sub: 'XLM-RoBERTa' },
  { key: 'svm',      label: 'SVM',                 sub: 'TF-IDF + SVM' },
  { key: 'lr',       label: 'Logistic Regression', sub: 'TF-IDF + LR' },
  { key: 'ensemble', label: 'Ensemble',             sub: 'All Models Combined' },
];

function ModelSelector({ value, onChange, label = 'Choose model:' }) {
  return (
    <div className="model-selector">
      <span className="model-selector-label">{label}</span>
      <div className="model-btn-group">
        {MODEL_OPTIONS.map((opt) => (
          <button
            key={opt.key}
            className={`model-btn ${value === opt.key ? 'active' : ''}`}
            onClick={() => onChange(opt.key)}
          >
            <span className="model-btn-name">{opt.label}</span>
            <span className="model-btn-sub">{opt.sub}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

export default ModelSelector;
