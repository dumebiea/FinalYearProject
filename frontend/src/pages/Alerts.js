import React, { useState, useEffect } from 'react';
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
import api from '../api';
import '../styles/Alerts.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
);

function Alerts({ user }) {
  const [regions, setRegions] = useState([]);
  const [selectedRegion, setSelectedRegion] = useState(null);
  const [chartData, setChartData] = useState(null);
  const [statusSummary, setStatusSummary] = useState(null);
  const [trendKeywords, setTrendKeywords] = useState([]);
  const [plainEnglishText, setPlainEnglishText] = useState('');
  const [chartPeriodLabel, setChartPeriodLabel] = useState('Past 30 Days');
  const [loading, setLoading] = useState(true);
  const [loadingForecast, setLoadingForecast] = useState(false);

  const LAGOS_REGIONS = [
    'Ikeja', 'Surulere', 'Lagos Island', 'Oshodi', 'Alimosho', 'Badagry', 'Epe',
  ];

  const generateReason = (regionName, negativePct) => {
    const reasons = [
      'Rising food price complaints',
      'Increasing hunger mentions',
      'Supply chain disruptions reported',
      'Market availability concerns',
      'Cost of living complaints',
      'Food shortage discussions',
      'Welfare program delays',
    ];
    const name = typeof regionName === 'string' ? regionName : '';
    const hash = (name.length > 0 ? name.charCodeAt(0) : 0) + Math.round(negativePct || 0);
    return reasons[hash % reasons.length];
  };

  // Load region list for the crisis grid + default region selection
  useEffect(() => {
    const loadRegionData = async () => {
      setLoading(true);
      try {
        const response = await api.get('/dashboard/heatmap', { params: { days: 30 } });
        const regionsList = (response.data.locations || []).sort(
          (a, b) => b.negative_pct - a.negative_pct
        );
        setRegions(regionsList);
        if (regionsList.length > 0) setSelectedRegion(regionsList[0].name);
      } catch (err) {
        console.error('Error loading region data:', err);
        setRegions([]);
      } finally {
        setLoading(false);
      }
    };
    loadRegionData();
  }, []);

  // Load trend + forecast + keywords whenever selected region changes
  useEffect(() => {
    if (!selectedRegion) return;

    const loadForecastData = async () => {
      setLoadingForecast(true);
      try {
        const [trendsRes, keywordsRes] = await Promise.all([
          api.get('/dashboard/trends', { params: { days: 90, location: selectedRegion } }),
          api.get('/dashboard/keywords', { params: { days: 7, sentiment: 'Negative' } }),
        ]);

        const allLabels   = trendsRes.data.labels   || [];
        const allNegative = trendsRes.data.negative  || [];
        const allPositive = trendsRes.data.positive  || [];
        const allNeutral  = trendsRes.data.neutral   || [];

        const forecastLabels = trendsRes.data.forecast_labels || [];
        const forecastValues = trendsRes.data.forecast_values || [];
        const forecastUpper  = trendsRes.data.forecast_upper  || [];
        const forecastLower  = trendsRes.data.forecast_lower  || [];
        const trendDirection = trendsRes.data.trend_direction || 'STABLE';
        const slope          = trendsRes.data.slope           || 0;

        // Show only last 30 days of the 90-day history
        const DISPLAY_DAYS = 30;
        const start = Math.max(0, allLabels.length - DISPLAY_DAYS);
        const displayLabels   = allLabels.slice(start);
        const displayNegative = allNegative.slice(start);
        const displayPositive = allPositive.slice(start);
        const displayNeutral  = allNeutral.slice(start);

        const actualDays = displayLabels.length;
        setChartPeriodLabel(actualDays < DISPLAY_DAYS ? `Past ${actualDays} Days` : 'Past 30 Days');

        const nForecast = forecastValues.length;

        // Historical lines padded with null for the forecast period
        const negHistData = [...displayNegative, ...Array(nForecast).fill(null)];
        const posHistData = [...displayPositive, ...Array(nForecast).fill(null)];
        const neuHistData = [...displayNeutral,  ...Array(nForecast).fill(null)];

        // Crisis markers: bigger + darker red on days where negative > 50
        const negPointRadius = [
          ...displayNegative.map(v => (v > 50 ? 8 : 3)),
          ...Array(nForecast).fill(0),
        ];
        const negPointColor = [
          ...displayNegative.map(v => (v > 50 ? '#7B241C' : '#C0392B')),
          ...Array(nForecast).fill('transparent'),
        ];
        const negPointBorder = [
          ...displayNegative.map(v => (v > 50 ? '#fff' : '#C0392B')),
          ...Array(nForecast).fill('transparent'),
        ];

        // 50% threshold line across the full x-axis
        const allChartLabels = [...displayLabels, ...forecastLabels];
        const thresholdData  = Array(allChartLabels.length).fill(50);

        const datasets = [
          // Threshold line
          {
            label: 'Crisis Threshold (50%)',
            data: thresholdData,
            borderColor: 'rgba(192, 57, 43, 0.35)',
            backgroundColor: 'transparent',
            borderWidth: 1,
            borderDash: [8, 4],
            pointRadius: 0,
            tension: 0,
            fill: false,
          },
          // Negative — main focus line
          {
            label: 'Negative',
            data: negHistData,
            borderColor: '#C0392B',
            backgroundColor: 'transparent',
            borderWidth: 2.5,
            tension: 0.3,
            fill: false,
            pointRadius: negPointRadius,
            pointBackgroundColor: negPointColor,
            pointBorderColor: negPointBorder,
            pointBorderWidth: 2,
          },
          // Positive
          {
            label: 'Positive',
            data: posHistData,
            borderColor: '#27AE60',
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            tension: 0.3,
            fill: false,
            pointRadius: 2,
            pointBackgroundColor: '#27AE60',
          },
          // Neutral
          {
            label: 'Neutral',
            data: neuHistData,
            borderColor: '#AAAAAA',
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            tension: 0.3,
            fill: false,
            pointRadius: 2,
            pointBackgroundColor: '#AAAAAA',
          },
        ];

        // 7-day forecast extension
        if (nForecast > 0) {
          const negForecastData = [
            ...Array(displayNegative.length).fill(null),
            ...forecastValues,
          ];

          datasets.push({
            label: '7-Day Prediction',
            data: negForecastData,
            borderColor: '#C0392B',
            backgroundColor: 'transparent',
            borderWidth: 2,
            borderDash: [5, 5],
            tension: 0.3,
            fill: false,
            pointRadius: [
              ...Array(displayNegative.length).fill(0),
              ...forecastValues.map(() => 4),
            ],
            pointBackgroundColor: '#E67E22',
            pointBorderColor: '#fff',
            pointBorderWidth: 1,
          });

          // Shaded uncertainty band
          if (forecastUpper.length > 0 && forecastLower.length > 0) {
            const upperData = [...Array(displayNegative.length).fill(null), ...forecastUpper];
            const lowerData = [...Array(displayNegative.length).fill(null), ...forecastLower];

            datasets.push({
              label: 'Prediction Range',
              data: upperData,
              borderColor: 'transparent',
              backgroundColor: 'rgba(230, 126, 34, 0.13)',
              borderWidth: 0,
              fill: '+1',
              pointRadius: 0,
              tension: 0.3,
            });
            datasets.push({
              label: '_lower',
              data: lowerData,
              borderColor: 'transparent',
              backgroundColor: 'rgba(230, 126, 34, 0.13)',
              borderWidth: 0,
              fill: false,
              pointRadius: 0,
              tension: 0.3,
            });
          }
        }

        setChartData({ labels: allChartLabels, datasets });

        // --- Status summary ---
        const recentNeg = displayNegative.filter(v => v > 0);
        const latestNeg = recentNeg.length > 0 ? recentNeg[recentNeg.length - 1] : 0;
        const crisisDays = displayNegative.filter(v => v > 50).length;

        let riskLevel = 'Low';
        let riskColor = '#27AE60';
        let currentMood = 'Calm';
        if (latestNeg > 50) {
          riskLevel = 'High';
          riskColor = '#C0392B';
          currentMood = 'Distressed';
        } else if (latestNeg > 35) {
          riskLevel = 'Medium';
          riskColor = '#E67E22';
          currentMood = 'Concerned';
        }

        setStatusSummary({
          currentMood,
          latestNeg: latestNeg.toFixed(1),
          riskLevel,
          riskColor,
          crisisDays,
          trendDirection,
          forecastAvailable: nForecast > 0,
          forecastEnd: nForecast > 0 ? forecastValues[nForecast - 1].toFixed(1) : null,
          forecastEndDate: forecastLabels.length > 0 ? forecastLabels[forecastLabels.length - 1] : null,
          hasSufficientData: trendsRes.data.has_sufficient_data,
          warning: trendsRes.data.forecast_warning,
        });

        // --- Plain English paragraph ---
        const periodLabel = actualDays < 30 ? `the past ${actualDays} days` : 'the past 30 days';
        let text = '';

        if (trendDirection === 'RISING') {
          text = `Negative sentiment in ${selectedRegion} has been rising over ${periodLabel}. `;
        } else if (trendDirection === 'FALLING') {
          text = `Negative sentiment in ${selectedRegion} has been declining over ${periodLabel}. `;
        } else {
          text = `Negative sentiment in ${selectedRegion} has remained relatively stable over ${periodLabel}. `;
        }

        if (latestNeg > 50) {
          text += `Right now, ${latestNeg.toFixed(1)}% of food-related posts are negative — above the crisis alert threshold of 50%. `;
        } else if (latestNeg > 35) {
          text += `Right now, ${latestNeg.toFixed(1)}% of food-related posts are negative — approaching the crisis alert level. `;
        } else {
          text += `Right now, ${latestNeg.toFixed(1)}% of food-related posts are negative, which is within the normal range. `;
        }

        if (nForecast > 0) {
          const endVal = parseFloat(forecastValues[nForecast - 1].toFixed(1));
          const endDate = forecastLabels[forecastLabels.length - 1];
          if (trendDirection === 'RISING' && endVal > 50) {
            text += `If this trend continues, the situation is expected to reach ${endVal}% by ${endDate}, which would signal a crisis. Immediate attention is recommended.`;
          } else if (trendDirection === 'RISING') {
            text += `If this trend continues, negative sentiment is projected to reach ${endVal}% by ${endDate}. Close monitoring is advised.`;
          } else if (trendDirection === 'FALLING') {
            text += `The situation is expected to improve, with sentiment projected to drop to ${endVal}% by ${endDate}.`;
          } else {
            text += `The situation is expected to remain around ${endVal}% over the next 7 days.`;
          }
        } else {
          text += 'There is not yet enough data to generate a reliable 7-day prediction for this region.';
        }

        setPlainEnglishText(text);

        // Keywords (top 5)
        setTrendKeywords((keywordsRes.data.keywords || []).slice(0, 5));

      } catch (err) {
        console.error('Error loading forecast data:', err);
        setChartData(null);
        setStatusSummary(null);
        setPlainEnglishText('');
        setTrendKeywords([]);
      } finally {
        setLoadingForecast(false);
      }
    };

    loadForecastData();
  }, [selectedRegion]);

  const getRiskColor = (negativePct) => {
    if (negativePct > 50) return '#C0392B';
    if (negativePct > 35) return '#E67E22';
    return '#27AE60';
  };

  const getRiskLevel = (negativePct) => {
    if (negativePct > 50) return 'High Risk';
    if (negativePct > 35) return 'Medium Risk';
    return 'Low Risk';
  };

  if (loading) {
    return <div className="alerts-loading">Loading crisis analysis...</div>;
  }

  const displayRegions = regions.length > 0
    ? regions
    : LAGOS_REGIONS.map(name => ({ name, negative_pct: null, total: null }));

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: {
          font: { size: 11 },
          padding: 12,
          usePointStyle: true,
          filter: (item) => !item.text.startsWith('_'),
        },
      },
      title: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx) => {
            if (ctx.dataset.label?.startsWith('_')) return null;
            if (ctx.dataset.label === 'Crisis Threshold (50%)') return null;
            if (ctx.parsed.y === null) return null;
            return `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}%`;
          },
        },
        filter: (item) => item.parsed.y !== null,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          callback: (value) => value + '%',
          font: { size: 11 },
        },
        title: {
          display: true,
          text: 'Share of Posts (%)',
          font: { size: 11 },
        },
      },
      x: {
        ticks: {
          maxTicksLimit: 10,
          font: { size: 10 },
          maxRotation: 30,
        },
      },
    },
  };

  return (
    <div className="alerts-container">
      {/* Page Header */}
      <div className="alerts-header">
        <h1>Sentiment Forecast and Crisis Prediction</h1>
        <p className="alerts-subtitle">Based on Lagos Twitter data analysis</p>
      </div>

      {/* Region selector */}
      <div className="region-selector-bar">
        <label>Viewing data for:</label>
        <select
          value={selectedRegion || ''}
          onChange={(e) => setSelectedRegion(e.target.value)}
        >
          {displayRegions.map((region) => (
            <option key={region.name} value={region.name}>
              {region.name}
            </option>
          ))}
        </select>
      </div>

      {/* ===== THREE-PART FORECAST SECTION ===== */}
      {loadingForecast ? (
        <div className="forecast-loading">Loading forecast data...</div>
      ) : (
        <>
          {/* PART 1: Past 30-Day Trend Chart */}
          <div className="forecast-block">
            <div className="forecast-block-header">
              <h2>What Has Been Happening</h2>
              <span className="period-badge">{chartPeriodLabel}</span>
            </div>
            <p className="forecast-block-desc">
              Share of food-related posts that are negative (red), positive (green), or neutral (grey).
              Large red dots mark days when the situation exceeded the crisis level.
              The dotted section shows the 7-day prediction.
            </p>
            <div className="trend-chart-wrapper">
              {chartData ? (
                <Line data={chartData} options={chartOptions} />
              ) : (
                <div className="no-chart-msg">No trend data is available for this region yet.</div>
              )}
            </div>
          </div>

          {/* PART 2: Current Status Summary */}
          {statusSummary && (
            <div className="forecast-block status-summary-block">
              <div className="forecast-block-header">
                <h2>What Is Happening Now</h2>
              </div>

              <div className="status-cards-row">
                <div className="status-card">
                  <div className="status-card-label">Current Mood</div>
                  <div className="status-card-value" style={{ color: statusSummary.riskColor }}>
                    {statusSummary.currentMood}
                  </div>
                  <div className="status-card-sub">{statusSummary.latestNeg}% of posts are negative</div>
                </div>

                <div
                  className="status-card risk-card"
                  style={{ borderTopColor: statusSummary.riskColor }}
                >
                  <div className="status-card-label">Crisis Risk Level</div>
                  <div className="status-card-value" style={{ color: statusSummary.riskColor }}>
                    {statusSummary.riskLevel}
                  </div>
                  {statusSummary.crisisDays > 0 && (
                    <div className="status-card-sub">
                      {statusSummary.crisisDays} crisis day{statusSummary.crisisDays !== 1 ? 's' : ''} in this period
                    </div>
                  )}
                  {statusSummary.crisisDays === 0 && (
                    <div className="status-card-sub">No crisis days recorded</div>
                  )}
                </div>

                <div className="status-card keywords-card">
                  <div className="status-card-label">Trending Topics</div>
                  <div className="keywords-list">
                    {trendKeywords.length > 0 ? (
                      trendKeywords.map((kw, i) => (
                        <span key={i} className="keyword-tag">{kw}</span>
                      ))
                    ) : (
                      <span className="status-card-sub">No keyword data available</span>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* PART 3: 7-Day Prediction — Plain English */}
          {plainEnglishText && (
            <div className="forecast-block prediction-block">
              <div className="forecast-block-header">
                <h2>What Is Likely to Happen</h2>
                <span className="period-badge">Next 7 Days</span>
              </div>

              {statusSummary?.warning && !statusSummary?.hasSufficientData && (
                <div className="data-note">
                  Note: Limited data available — the prediction may be less accurate than usual.
                </div>
              )}

              <div className="plain-english-box">
                <p>{plainEnglishText}</p>
              </div>

              {statusSummary?.forecastAvailable && (
                <div className="forecast-horizon-note">
                  The shaded orange band in the chart above shows the range within which the actual figure is likely to fall.
                </div>
              )}
            </div>
          )}
        </>
      )}

      {/* Crisis Risk by Region */}
      <div className="alerts-section region-section">
        <div className="section-header">
          <h2>Crisis Risk by Region</h2>
          <p className="section-desc">Ranked by current negative sentiment — click a region to view its forecast above</p>
        </div>

        <div className="region-grid">
          {displayRegions.map((region) => (
            <div
              key={region.name}
              className={`region-card ${selectedRegion === region.name ? 'selected' : ''}`}
              onClick={() => setSelectedRegion(region.name)}
              style={{
                borderLeftColor: region.negative_pct !== null
                  ? getRiskColor(region.negative_pct)
                  : '#999',
              }}
            >
              <div className="region-header">
                <h3>{region.name}</h3>
                {region.negative_pct !== null && (
                  <div className="region-sentiment">{region.negative_pct.toFixed(1)}%</div>
                )}
              </div>

              {region.negative_pct !== null && (
                <>
                  <div className="region-risk">
                    <span
                      className="risk-badge"
                      style={{
                        background: getRiskColor(region.negative_pct) + '22',
                        color: getRiskColor(region.negative_pct),
                        border: `1px solid ${getRiskColor(region.negative_pct)}`,
                      }}
                    >
                      {getRiskLevel(region.negative_pct)}
                    </span>
                  </div>
                  <div className="region-reason">
                    <p>{generateReason(region.name, region.negative_pct)}</p>
                  </div>
                  <div className="region-tweets">
                    <small>{region.total} tweets analyzed</small>
                  </div>
                </>
              )}

              {region.negative_pct === null && (
                <div className="region-reason">
                  <p style={{ color: '#999' }}>No data available</p>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default Alerts;
