import React, { useState } from 'react';
import { uploadCSV, searchTweets } from '../api';
import ModelSelector from '../components/ModelSelector';

const LAGOS_LGAS = [
  'All Lagos',
  'Oshodi',
  'Ajeromi-Ifelodun',
  'Ojo',
  'Eti-Osa',
  'Lagos Island',
  'Lagos Mainland',
  'Alimosho',
  'Mushin',
  'Surulere',
  'Kosofe',
  'Ikorodu',
  'Shomolu',
  'Apapa',
  'Ikeja',
  'Agege',
  'Badagry',
  'Epe',
  'Ifako-Ijaiye',
  'Somolu',
  'Baruwa'
];

function Upload() {
  const [activeTab, setActiveTab] = useState('csv');
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [fileSize, setFileSize] = useState(0);
  const [keywords, setKeywords] = useState('');
  const [location, setLocation] = useState('All Lagos');
  const [selectedModel, setSelectedModel] = useState('bert');
  const [loading, setLoading] = useState(false);
  const [csvResults, setCsvResults] = useState(null);
  const [searchResults, setSearchResults] = useState(null);
  const [csvPreview, setCsvPreview] = useState([]);
  const [searchPreview, setSearchPreview] = useState([]);
  const [error, setError] = useState('');
  const [searchPage, setSearchPage] = useState(1);
  const RESULTS_PER_PAGE = 20;

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type === 'text/csv') {
      setFile(selectedFile);
      setFileName(selectedFile.name);
      setFileSize((selectedFile.size / 1024).toFixed(2));
      setError('');
      setCsvResults(null);

      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target.result;
        const lines = text.split('\n').slice(0, 6);
        setCsvPreview(lines);
      };
      reader.readAsText(selectedFile);
    } else {
      setError('Please select a valid CSV file');
      setFile(null);
      setCsvPreview([]);
    }
  };

  const handleCsvUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await uploadCSV(formData, selectedModel);
      setCsvResults(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Upload failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleKeywordSearch = async () => {
    if (!keywords.trim()) {
      setError('Please enter at least one keyword');
      return;
    }
    setLoading(true);
    setError('');
    setSearchPage(1);
    try {
      const keywordArray = keywords.split(',').map(k => k.trim()).filter(k => k);
      const response = await searchTweets(keywordArray, location, selectedModel);
      setSearchResults(response.data);
      setSearchPreview(response.data.results?.slice(0, RESULTS_PER_PAGE) || []);
    } catch (err) {
      setError(err.response?.data?.error || 'Search failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const getSentimentBadge = (sentiment) => {
    const upper = (sentiment || '').toUpperCase();
    const map = {
      'NEGATIVE': { color: '#C0392B', emoji: '🔴' },
      'POSITIVE': { color: '#27AE60', emoji: '🟢' },
      'NEUTRAL':  { color: '#888888', emoji: '⚪' },
    };
    const cfg = map[upper] || map['NEUTRAL'];
    return (
      <span style={{ color: cfg.color, fontWeight: 'bold' }}>
        {cfg.emoji} {upper}
      </span>
    );
  };

  const paginatedResults = searchResults?.results?.slice(
    (searchPage - 1) * RESULTS_PER_PAGE,
    searchPage * RESULTS_PER_PAGE
  ) || [];
  const totalSearchPages = Math.ceil((searchResults?.results?.length || 0) / RESULTS_PER_PAGE);
  const isEnsembleSearch = searchResults?.model_used === 'Ensemble';

  return (
    <div className="upload-container">
      <div className="upload-header">
        <h1>Upload & Search Tweets</h1>
      </div>

      <div className="tab-buttons">
        <button
          className={`tab-btn ${activeTab === 'csv' ? 'active' : ''}`}
          onClick={() => setActiveTab('csv')}
        >
          CSV Upload
        </button>
        <button
          className={`tab-btn ${activeTab === 'search' ? 'active' : ''}`}
          onClick={() => setActiveTab('search')}
        >
          Keyword Search
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      {/* CSV Upload Tab */}
      {activeTab === 'csv' && (
        <div className="tab-content">
          <div className="card upload-card">
            <div className="instructions">
              Upload a CSV file containing tweets. The file must have a column named tweet_text, tweet, text, or content.
            </div>

            <div className="file-picker-section">
              <input
                type="file"
                id="csv-file"
                accept=".csv"
                onChange={handleFileChange}
                className="file-input"
              />
              {fileName && (
                <div className="file-info">
                  <span className="file-name">📄 {fileName}</span>
                  <span className="file-size">({fileSize} KB)</span>
                </div>
              )}
            </div>

            {csvPreview.length > 0 && (
              <div className="preview-section">
                <h4>Preview (first 5 rows)</h4>
                <table className="preview-table">
                  <tbody>
                    {csvPreview.map((row, idx) => (
                      <tr key={idx}><td>{row}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <ModelSelector
              value={selectedModel}
              onChange={setSelectedModel}
              label="Classification model:"
            />

            <button
              className="primary-btn full-width"
              onClick={handleCsvUpload}
              disabled={!file || loading}
            >
              {loading ? (
                <><span className="spinner"></span> Processing tweets...</>
              ) : (
                'Classify All Tweets'
              )}
            </button>
          </div>

          {csvResults && (
            <div className="card results-card">
              <div className="summary-stats">
                <div className="stat-card">
                  <div className="stat-label">Total Classified</div>
                  <div className="stat-value">{csvResults.total || 0}</div>
                </div>
                <div className="stat-card negative">
                  <div className="stat-label">Negative</div>
                  <div className="stat-value">{csvResults.negative || 0}</div>
                </div>
                <div className="stat-card neutral">
                  <div className="stat-label">Neutral</div>
                  <div className="stat-value">{csvResults.neutral || 0}</div>
                </div>
                <div className="stat-card positive">
                  <div className="stat-label">Positive</div>
                  <div className="stat-value">{csvResults.positive || 0}</div>
                </div>
              </div>

              <div className="success-message">
                ✅ {csvResults.total || 0} tweets classified using <strong>{csvResults.model_used}</strong> and saved to database
              </div>

              {/* Ensemble per-model breakdown */}
              {csvResults.ensemble_breakdown && (
                <div className="ensemble-breakdown-table">
                  <h4>Ensemble — Per-Model Vote Breakdown</h4>
                  <table className="results-table">
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Negative</th>
                        <th>Neutral</th>
                        <th>Positive</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(csvResults.ensemble_breakdown).map(([mname, counts]) => (
                        <tr key={mname}>
                          <td><strong>{mname}</strong></td>
                          <td style={{ color: '#C0392B', fontWeight: 600 }}>{counts.Negative}</td>
                          <td style={{ color: '#888888', fontWeight: 600 }}>{counts.Neutral}</td>
                          <td style={{ color: '#27AE60', fontWeight: 600 }}>{counts.Positive}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              <a href="/dashboard" className="view-dashboard-btn">
                View Dashboard →
              </a>
            </div>
          )}
        </div>
      )}

      {/* Keyword Search Tab */}
      {activeTab === 'search' && (
        <div className="tab-content">
          <div className="card search-card">
            <div className="instructions">
              Enter keywords to search for relevant tweets in the dataset. Results will be classified and added to the dashboard.
            </div>

            <div className="search-form">
              <div className="form-group">
                <label htmlFor="keywords">Keywords (comma-separated)</label>
                <input
                  type="text"
                  id="keywords"
                  value={keywords}
                  onChange={(e) => setKeywords(e.target.value)}
                  placeholder="hunger, food price, rice, scarcity"
                  className="form-input"
                />
              </div>

              <div className="form-group">
                <label htmlFor="location">Location</label>
                <select
                  id="location"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  className="form-input"
                >
                  {LAGOS_LGAS.map(lga => (
                    <option key={lga} value={lga}>{lga}</option>
                  ))}
                </select>
              </div>

              <ModelSelector
                value={selectedModel}
                onChange={setSelectedModel}
                label="Classification model:"
              />

              <button
                className="primary-btn full-width"
                onClick={handleKeywordSearch}
                disabled={!keywords.trim() || loading}
              >
                {loading ? (
                  <><span className="spinner"></span> Searching...</>
                ) : (
                  'Search and Classify'
                )}
              </button>
            </div>
          </div>

          {searchResults && searchResults.results && searchResults.results.length > 0 && (
            <div className="card results-card">
              <div className="results-header">
                {searchResults.results.length} tweets found and classified
                {searchResults.model_used && (
                  <span className="results-model-tag"> · {searchResults.model_used}</span>
                )}
              </div>

              {/* Single-model results table */}
              {!isEnsembleSearch && (
                <table className="results-table">
                  <thead>
                    <tr>
                      <th>Tweet Text</th>
                      <th>Predicted Label</th>
                      <th>Confidence</th>
                      <th>Model</th>
                      <th>Location</th>
                    </tr>
                  </thead>
                  <tbody>
                    {paginatedResults.map((result, idx) => (
                      <tr key={idx}>
                        <td className="tweet-cell">{result.tweet?.substring(0, 50)}...</td>
                        <td>{getSentimentBadge(result.label)}</td>
                        <td>{(parseFloat(result.confidence) * 100).toFixed(1)}%</td>
                        <td style={{ fontSize: '0.82rem', color: '#555' }}>{result.model_used}</td>
                        <td>{result.location || location}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}

              {/* Ensemble results table */}
              {isEnsembleSearch && (
                <>
                  <table className="results-table ensemble-results-table">
                    <thead>
                      <tr>
                        <th>Tweet Text</th>
                        <th>NaijaSenti</th>
                        <th>SVM</th>
                        <th>LR</th>
                        <th>Final Decision</th>
                        <th>Votes</th>
                        <th>Location</th>
                      </tr>
                    </thead>
                    <tbody>
                      {paginatedResults.map((result, idx) => {
                        const ind = result.individual || [];
                        const getInd = (name) => ind.find(i => i.model.includes(name));
                        const bert = getInd('NaijaSenti');
                        const svm  = getInd('SVM');
                        const lr   = getInd('Logistic');
                        return (
                          <tr key={idx}>
                            <td className="tweet-cell">{result.tweet?.substring(0, 45)}...</td>
                            <td>{bert ? getSentimentBadge(bert.label) : '—'}</td>
                            <td>{svm  ? getSentimentBadge(svm.label)  : '—'}</td>
                            <td>{lr   ? getSentimentBadge(lr.label)   : '—'}</td>
                            <td>{getSentimentBadge(result.label)}</td>
                            <td style={{ textAlign: 'center', fontWeight: 600, color: '#185FA5' }}>
                              {result.votes}/{result.total}
                            </td>
                            <td>{result.location || location}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>

                  {/* Ensemble aggregate summary */}
                  <div className="ensemble-summary-bar">
                    <span>Ensemble Final Results — </span>
                    <span style={{ color: '#C0392B', fontWeight: 700 }}>
                      Negative: {searchResults.negative}
                    </span>
                    <span> · </span>
                    <span style={{ color: '#888', fontWeight: 700 }}>
                      Neutral: {searchResults.neutral}
                    </span>
                    <span> · </span>
                    <span style={{ color: '#27AE60', fontWeight: 700 }}>
                      Positive: {searchResults.positive}
                    </span>
                  </div>
                </>
              )}

              {totalSearchPages > 1 && (
                <div className="pagination">
                  <button
                    onClick={() => setSearchPage(prev => Math.max(1, prev - 1))}
                    disabled={searchPage === 1}
                  >
                    Previous
                  </button>
                  <span>Page {searchPage} of {totalSearchPages}</span>
                  <button
                    onClick={() => setSearchPage(prev => Math.min(totalSearchPages, prev + 1))}
                    disabled={searchPage === totalSearchPages}
                  >
                    Next
                  </button>
                </div>
              )}
            </div>
          )}

          {searchResults && searchResults.results && searchResults.results.length === 0 && (
            <div className="card empty-state">
              <p>No tweets found matching your search criteria. Try different keywords or locations.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default Upload;
