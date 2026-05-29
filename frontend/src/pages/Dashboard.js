import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import {
  getDashboardSummary,
  getDashboardTrends,
  getDashboardKeywords,
  getDashboardHeatmap,
  getActiveAlerts,
} from '../api';
import api from '../api';
import AlertBanner from '../components/AlertBanner';
import SentimentPieChart from '../components/SentimentPieChart';
import SentimentTrendChart from '../components/SentimentTrendChart';
import KeywordBarChart from '../components/KeywordBarChart';
import TweetVolumeChart from '../components/TweetVolumeChart';
import LocationHeatmap from '../components/LocationHeatmap';
import Navbar from '../components/Navbar';
import LiveStream from '../components/LiveStream';
import { useTheme } from '../contexts/ThemeContext';
import { generateDashboardPDF } from '../utils/generatePDF';
import { playAlertChime } from '../utils/alertSound';
import '../styles/Dashboard.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

function ChartLoading() {
  return (
    <div className="chart-loading">
      <div className="chart-spinner" />
      <p>Loading...</p>
    </div>
  );
}

function ChartEmpty() {
  return (
    <div className="chart-empty">
      <p>No data available for this date range</p>
    </div>
  );
}

function Dashboard() {
  const [summary, setSummary] = useState(null);
  const [trends, setTrends] = useState(null);
  const [keywords, setKeywords] = useState({ keywords: [], frequencies: [] });
  const [heatmap, setHeatmap] = useState([]);
  const [activeAlerts, setActiveAlerts] = useState([]);
  const [forecastTrends, setForecastTrends] = useState(null);
  const [forecastInfo, setForecastInfo] = useState(null);
  const [forecastMessage, setForecastMessage] = useState(null);
  const [chartsLoading, setChartsLoading] = useState(true);
  const [filters, setFilters] = useState({ days: 30, location: 'All Lagos' });
  const [alertReady, setAlertReady] = useState(false);
  const [expandedChart, setExpandedChart] = useState(null);
  const [pdfLoading, setPdfLoading] = useState(false);
  const monitoringStartRef = useRef(Date.now());
  const consecutiveNegRef = useRef(0);
  const lastSeenTimestampRef = useRef(new Date().toISOString());
  const seenAlertIdsRef = useRef(null); // null = first load, skip sound on initial paint
  const { isDark } = useTheme();

  const openChart = (title, node) => setExpandedChart({ title, node });
  const closeChart = () => setExpandedChart(null);

  // useCallback ensures loadDashboardData always captures the current filters value,
  // and the useEffect below re-runs whenever filters change.
  const loadDashboardData = useCallback(async () => {
    setChartsLoading(true);
    try {
      const [summaryRes, trendsRes, keywordsRes, heatmapRes, alertsRes] = await Promise.all([
        getDashboardSummary(filters.days, filters.location),
        getDashboardTrends(filters.days, filters.location),
        getDashboardKeywords(filters.days, 'Negative'),
        getDashboardHeatmap(filters.days),
        getActiveAlerts(),
      ]);

      setSummary(summaryRes.data || {
        total_tweets: 0, negative_pct: 0, neutral_pct: 0,
        positive_pct: 0, avg_confidence: 0, active_alerts: 0, has_data: false,
      });

      setTrends(trendsRes.data || {
        labels: [], negative: [], neutral: [], positive: [],
      });

      setKeywords({
        keywords: keywordsRes.data?.keywords || [],
        frequencies: keywordsRes.data?.frequencies || [],
      });
      setHeatmap(heatmapRes.data?.locations || heatmapRes.data || []);
      setActiveAlerts(alertsRes.data?.alerts || alertsRes.data || []);

      // Forecast — uses the currently selected date range
      try {
        const forecastRes = await api.get('/dashboard/trends', {
          params: { days: filters.days, location: filters.location },
        });

        if (forecastRes.data.forecast_values && forecastRes.data.forecast_values.length > 0) {
          const labels = forecastRes.data.labels || [];
          const negativeData = forecastRes.data.negative || [];
          const forecastLabels = forecastRes.data.forecast_labels || [];
          const forecastValues = forecastRes.data.forecast_values || [];
          const forecastUpper = forecastRes.data.forecast_upper || [];
          const forecastLower = forecastRes.data.forecast_lower || [];
          const trendDirection = forecastRes.data.trend_direction || 'STABLE';
          const slope = forecastRes.data.slope || 0;

          const allLabels = [...labels, ...forecastLabels];

          const datasets = [
            {
              label: 'Negative Sentiment %',
              data: negativeData,
              borderColor: '#C0392B',
              backgroundColor: 'rgba(192, 57, 43, 0.1)',
              borderWidth: 2,
              tension: 0.4,
              fill: false,
              pointRadius: 2,
              pointBackgroundColor: '#C0392B',
              pointBorderColor: '#fff',
              pointBorderWidth: 1,
            },
          ];

          if (forecastValues.length > 0) {
            const paddedForecast = [
              ...Array(negativeData.length).fill(null),
              ...forecastValues,
            ];

            datasets.push({
              label: '7-Day Forecast',
              data: paddedForecast,
              borderColor: '#E67E22',
              backgroundColor: 'rgba(230, 126, 34, 0.1)',
              borderWidth: 2,
              borderDash: [5, 5],
              tension: 0.4,
              fill: false,
              pointRadius: 3,
              pointBackgroundColor: '#E67E22',
              pointBorderColor: '#fff',
              pointBorderWidth: 1,
            });

            if (forecastUpper && forecastLower) {
              const paddedUpper = [
                ...Array(negativeData.length).fill(null),
                ...forecastUpper,
              ];
              const paddedLower = [
                ...Array(negativeData.length).fill(null),
                ...forecastLower,
              ];

              datasets.push({
                label: '95% Confidence Interval',
                data: paddedUpper,
                borderColor: 'transparent',
                backgroundColor: 'rgba(230, 126, 34, 0.15)',
                borderWidth: 0,
                fill: true,
                pointRadius: 0,
                tension: 0.4,
              });

              datasets.push({
                label: 'Confidence Lower',
                data: paddedLower,
                borderColor: 'transparent',
                backgroundColor: 'rgba(230, 126, 34, 0.15)',
                borderWidth: 0,
                fill: '-1',
                pointRadius: 0,
                tension: 0.4,
              });
            }
          }

          setForecastTrends({ labels: allLabels, datasets });

          let trendEmoji = '➡️';
          let trendText = 'STABLE';
          if (trendDirection === 'RISING') { trendEmoji = '📈'; trendText = 'RISING'; }
          else if (trendDirection === 'FALLING') { trendEmoji = '📉'; trendText = 'FALLING'; }

          setForecastInfo({
            trendEmoji, trendText, trendDirection, slope,
            warning: forecastRes.data.forecast_warning,
          });
          setForecastMessage(null);
        } else {
          setForecastTrends(null);
          setForecastInfo(null);
          setForecastMessage(
            forecastRes.data.forecast_message ||
            'Not enough data points to generate a forecast. Please select a wider date range.'
          );
        }
      } catch (forecastError) {
        console.error('Error loading forecast:', forecastError);
        setForecastTrends(null);
        setForecastInfo(null);
        setForecastMessage('Unable to load forecast data.');
      }
    } catch (error) {
      console.error('Dashboard load failed:', error);
      setSummary({
        total_tweets: 0, negative_pct: 0, neutral_pct: 0,
        positive_pct: 0, avg_confidence: 0, active_alerts: 0, has_data: false,
      });
      setTrends({ labels: [], negative: [], neutral: [], positive: [] });
      setKeywords({ keywords: [], frequencies: [] });
      setHeatmap([]);
    } finally {
      setChartsLoading(false);
    }
  }, [filters]);

  useEffect(() => {
    loadDashboardData();
  }, [loadDashboardData]);

  // Alert gate: only unlock after 2 min of monitoring AND ≥10 consecutive negative tweets
  useEffect(() => {
    if (alertReady) return;

    const pollStream = async () => {
      try {
        const res = await api.get('/dashboard/stream-status');
        const tweets = res.data?.recent_tweets || [];

        const newTweets = tweets.filter(
          (t) => t.created_at > lastSeenTimestampRef.current
        );

        if (newTweets.length > 0) {
          newTweets.sort((a, b) => a.created_at.localeCompare(b.created_at));
          lastSeenTimestampRef.current = newTweets[newTweets.length - 1].created_at;

          for (const tweet of newTweets) {
            if (tweet.label === 'Negative') {
              consecutiveNegRef.current += 1;
            } else {
              consecutiveNegRef.current = 0;
            }
          }

          const elapsed = Date.now() - monitoringStartRef.current;
          if (elapsed >= 2 * 60 * 1000 && consecutiveNegRef.current >= 10) {
            setAlertReady(true);
          }
        }
      } catch (_) {
        // Stream errors don't affect the rest of the dashboard
      }
    };

    const interval = setInterval(pollStream, 10000);
    return () => clearInterval(interval);
  }, [alertReady]);

  // Play chime when a NEW crisis alert appears (skip on first load)
  useEffect(() => {
    const crisisAlerts = activeAlerts.filter((a) => a.alert_type === 'crisis');
    const currentIds = new Set(crisisAlerts.map((a) => a.alert_id));

    if (seenAlertIdsRef.current === null) {
      // First load — record what already exists without playing sound
      seenAlertIdsRef.current = currentIds;
      return;
    }

    const hasNew = [...currentIds].some((id) => !seenAlertIdsRef.current.has(id));
    if (hasNew) playAlertChime();
    seenAlertIdsRef.current = currentIds;
  }, [activeAlerts]);

  // Close expanded chart on Escape key
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') closeChart(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const handleFilterChange = (field, value) => {
    setFilters((prev) => ({ ...prev, [field]: value }));
  };

  const handleExportPDF = async () => {
    setPdfLoading(true);
    try {
      let recentTweets = [];
      try {
        const streamRes = await api.get('/dashboard/stream-status');
        recentTweets = streamRes.data?.recent_tweets || [];
      } catch (_) { /* stream unavailable — skip tweets section */ }

      await generateDashboardPDF({
        summary,
        filters,
        isDark,
        recentTweets,
      });
    } finally {
      setPdfLoading(false);
    }
  };

  const volumes = trends?.labels?.map((_, index) => {
    const neg = Number(trends.negative?.[index] ?? 0);
    const neu = Number(trends.neutral?.[index] ?? 0);
    const pos = Number(trends.positive?.[index] ?? 0);
    return neg + neu + pos;
  }) || [];

  // Per-chart data availability flags (only meaningful when not loading)
  const hasPieData = !chartsLoading && (summary?.total_tweets ?? 0) > 0;
  const hasTrendsData = !chartsLoading && trends?.negative?.some((v) => v > 0);
  const hasKeywordsData = !chartsLoading && keywords.keywords.length > 0;
  const hasVolumeData = !chartsLoading && volumes.some((v) => v > 0);
  const hasHeatmapData = !chartsLoading && heatmap.length > 0;

  return (
    <div className="dashboard">
      {/* Enlarged chart modal */}
      {expandedChart && (
        <div className="chart-modal-overlay" onClick={closeChart}>
          <div className="chart-modal" onClick={(e) => e.stopPropagation()}>
            <div className="chart-modal-header">
              <h3>{expandedChart.title}</h3>
              <button className="chart-modal-close" onClick={closeChart}>✕</button>
            </div>
            <div className="chart-modal-body">
              {expandedChart.node}
            </div>
          </div>
        </div>
      )}

      <AlertBanner alerts={activeAlerts} ready={alertReady} />
      <LiveStream />

      <div className="dashboard-header">
        <div>
          <h1>Food Crisis Detection Dashboard</h1>
          <p className="dashboard-subtitle">Track Lagos sentiment and alerts in real time</p>
        </div>
        <div className="dashboard-header-right">
          <button
            className="pdf-export-btn"
            onClick={handleExportPDF}
            disabled={pdfLoading || chartsLoading}
            title="Download a PDF report of the current dashboard"
          >
            {pdfLoading ? (
              <span className="pdf-btn-inner">
                <span className="pdf-spinner" /> Generating...
              </span>
            ) : (
              <span className="pdf-btn-inner">
                ⬇ Download PDF Report
              </span>
            )}
          </button>
          <div className="filters">
          <label>
            Days
            <select
              value={filters.days}
              onChange={(e) => handleFilterChange('days', Number(e.target.value))}
              disabled={chartsLoading}
            >
              <option value={7}>7 days</option>
              <option value={14}>14 days</option>
              <option value={30}>30 days</option>
              <option value={60}>60 days</option>
              <option value={90}>90 days</option>
            </select>
          </label>
          <label>
            Location
            <select
              value={filters.location}
              onChange={(e) => handleFilterChange('location', e.target.value)}
              disabled={chartsLoading}
            >
              <option value="All Lagos">All Lagos</option>
              <option value="Lagos Island">Lagos Island</option>
              <option value="Ikeja">Ikeja</option>
              <option value="Surulere">Surulere</option>
            </select>
          </label>
          {chartsLoading && <span className="filter-updating">Updating...</span>}
        </div>
        </div>
      </div>

      {/* Summary cards — always visible; show dashes while loading */}
      <div className="summary-cards">
        <div className={`card${chartsLoading ? ' card-loading' : ''}`}>
          <h3>Total Tweets</h3>
          <div className="metric">
            {chartsLoading ? '—' : (summary?.total_tweets?.toLocaleString() ?? 0)}
          </div>
        </div>
        <div className={`card negative-card${chartsLoading ? ' card-loading' : ''}`}>
          <h3>Negative %</h3>
          <div className="metric">
            {chartsLoading ? '—' : `${summary?.negative_pct ?? 0}%`}
          </div>
        </div>
        <div className={`card positive-card${chartsLoading ? ' card-loading' : ''}`}>
          <h3>Positive %</h3>
          <div className="metric">
            {chartsLoading ? '—' : `${summary?.positive_pct ?? 0}%`}
          </div>
        </div>
        <div className={`card warning-card${chartsLoading ? ' card-loading' : ''}`}>
          <h3>Active Alerts</h3>
          <div className="metric">
            {chartsLoading ? '—' : (summary?.active_alerts ?? 0)}
          </div>
        </div>
      </div>

      <div className="charts-row">
        <div
          className={`card chart-card${hasPieData ? ' expandable-card' : ''}`}
          onClick={() => hasPieData && openChart('Sentiment Distribution',
            <SentimentPieChart
              negative={summary.negative_pct ?? 0}
              neutral={summary.neutral_pct ?? 0}
              positive={summary.positive_pct ?? 0}
            />
          )}
        >
          <h3>Sentiment Distribution</h3>
          {chartsLoading ? (
            <ChartLoading />
          ) : !hasPieData ? (
            <ChartEmpty />
          ) : (
            <SentimentPieChart
              key={`pie-${filters.days}-${filters.location}`}
              negative={summary.negative_pct ?? 0}
              neutral={summary.neutral_pct ?? 0}
              positive={summary.positive_pct ?? 0}
            />
          )}
        </div>

        <div
          id="pdf-chart-trend"
          className={`card chart-card${hasTrendsData ? ' expandable-card' : ''}`}
          onClick={() => hasTrendsData && openChart('Sentiment Trend Over Time',
            <SentimentTrendChart
              labels={trends.labels || []}
              negative={trends.negative || []}
              neutral={trends.neutral || []}
              positive={trends.positive || []}
            />
          )}
        >
          <h3>Sentiment Trend Over Time</h3>
          {chartsLoading ? (
            <ChartLoading />
          ) : !hasTrendsData ? (
            <ChartEmpty />
          ) : (
            <SentimentTrendChart
              key={`trend-${filters.days}-${filters.location}`}
              labels={trends.labels || []}
              negative={trends.negative || []}
              neutral={trends.neutral || []}
              positive={trends.positive || []}
            />
          )}
        </div>
      </div>

      <div className="charts-row">
        <div
          id="pdf-chart-keywords"
          className={`card chart-card${hasKeywordsData ? ' expandable-card' : ''}`}
          onClick={() => hasKeywordsData && openChart('Top Keywords by Frequency',
            <KeywordBarChart
              keywords={keywords.keywords}
              frequencies={keywords.frequencies}
            />
          )}
        >
          <h3>Top Keywords by Frequency</h3>
          {chartsLoading ? (
            <ChartLoading />
          ) : !hasKeywordsData ? (
            <ChartEmpty />
          ) : (
            <KeywordBarChart
              key={`keywords-${filters.days}-${filters.location}`}
              keywords={keywords.keywords}
              frequencies={keywords.frequencies}
            />
          )}
        </div>

        <div
          id="pdf-chart-volume"
          className={`card chart-card${hasVolumeData ? ' expandable-card' : ''}`}
          onClick={() => hasVolumeData && openChart('Tweet Volume Over Time',
            <TweetVolumeChart
              labels={trends.labels || []}
              volumes={volumes}
            />
          )}
        >
          <h3>Tweet Volume Over Time</h3>
          {chartsLoading ? (
            <ChartLoading />
          ) : !hasVolumeData ? (
            <ChartEmpty />
          ) : (
            <TweetVolumeChart
              key={`volume-${filters.days}-${filters.location}`}
              labels={trends.labels || []}
              volumes={volumes}
            />
          )}
        </div>
      </div>

      <div
        className={`heatmap-card card${hasHeatmapData ? ' expandable-card' : ''}`}
        onClick={() => hasHeatmapData && openChart('Sentiment by Lagos LGA',
          <LocationHeatmap locations={heatmap} />
        )}
      >
        <h3>Sentiment by Lagos LGA</h3>
        {chartsLoading ? (
          <ChartLoading />
        ) : !hasHeatmapData ? (
          <ChartEmpty />
        ) : (
          <LocationHeatmap
            key={`heatmap-${filters.days}-${filters.location}`}
            locations={heatmap}
          />
        )}
      </div>

      {/* 90-Day Trend + 7-Day Forecast */}
      {!chartsLoading && forecastTrends && (
        <div className="card chart-card">
          <h3>
            {filters.days}-Day Trend &amp; 7-Day Forecast
            {forecastInfo && (
              <span style={{ fontSize: '0.85rem', fontWeight: 400, color: '#555', marginLeft: 14 }}>
                {forecastInfo.trendEmoji} {forecastInfo.trendText}
                &nbsp;·&nbsp; slope: {forecastInfo.slope > 0 ? '+' : ''}{forecastInfo.slope}% / day
              </span>
            )}
          </h3>
          {forecastInfo?.warning && (
            <div style={{ fontSize: '0.82rem', color: '#E67E22', marginBottom: 8 }}>
              ⚠️ {forecastInfo.warning}
            </div>
          )}
          <Line
            key={`forecast-${filters.location}`}
            data={forecastTrends}
            options={{
              responsive: true,
              maintainAspectRatio: true,
              plugins: {
                legend: {
                  display: true, position: 'top',
                  labels: { font: { size: 11 }, padding: 14, usePointStyle: true },
                },
                title: { display: false },
              },
              scales: {
                y: {
                  beginAtZero: true, max: 100,
                  ticks: { callback: (v) => v + '%' },
                  title: { display: true, text: 'Negative Sentiment %' },
                },
                x: { title: { display: true, text: 'Date' } },
              },
            }}
          />
        </div>
      )}

      {!chartsLoading && !forecastTrends && (
        <div className="card chart-empty" style={{ padding: '24px' }}>
          <p>
            {forecastMessage ?? 'Not enough data points to generate a forecast. Please select a wider date range.'}
          </p>
        </div>
      )}
    </div>
  );
}

export default Dashboard;
