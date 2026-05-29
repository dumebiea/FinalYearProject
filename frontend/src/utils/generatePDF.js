import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

/* ─── helpers ──────────────────────────────────────────────── */

const BRAND_BLUE  = [24, 95, 165];
const NEGATIVE_R  = [192, 57, 43];
const WARNING_R   = [230, 126, 34];
const POSITIVE_R  = [39, 174, 96];
const NEUTRAL_R   = [100, 100, 100];

const PAGE_W = 210; // A4 mm
const PAGE_H = 297;
const MARGIN  = 18;
const CONTENT_W = PAGE_W - MARGIN * 2;

function riskLevel(negativePct) {
  if (negativePct >= 50) return { label: 'CRISIS', color: NEGATIVE_R };
  if (negativePct >= 35) return { label: 'WARNING', color: WARNING_R };
  return { label: 'LOW', color: POSITIVE_R };
}

function modelLabel(model) {
  const map = {
    bert: 'NaijaSenti (XLM-RoBERTa BERT)',
    svm: 'Support Vector Machine (SVM)',
    lr: 'Logistic Regression',
    ensemble: 'Ensemble (BERT + SVM + LR)',
  };
  return map[model] || model;
}

function plainEnglishSummary(summary, filters) {
  const neg   = summary?.negative_pct ?? 0;
  const pos   = summary?.positive_pct ?? 0;
  const neu   = summary?.neutral_pct ?? 0;
  const total = summary?.total_tweets ?? 0;
  const risk  = riskLevel(neg);
  const loc   = filters?.location || 'All Lagos';
  const days  = filters?.days || 30;

  let opening = '';
  if (risk.label === 'CRISIS') {
    opening = `Over the past ${days} days in ${loc}, the system has detected a CRITICAL level of food-related distress on Twitter. `;
  } else if (risk.label === 'WARNING') {
    opening = `Over the past ${days} days in ${loc}, the system has detected an ELEVATED level of food-related concern on Twitter. `;
  } else {
    opening = `Over the past ${days} days in ${loc}, food-related sentiment on Twitter has been relatively stable. `;
  }

  const breakdown = `Out of ${total.toLocaleString()} tweets analysed, ${neg}% were classified as negative, ${neu}% as neutral, and ${pos}% as positive. `;

  let action = '';
  if (risk.label === 'CRISIS') {
    action = 'Immediate attention from food security authorities is recommended. Coordinated intervention and public communication strategies should be considered urgently.';
  } else if (risk.label === 'WARNING') {
    action = 'Authorities should monitor the situation closely and prepare contingency measures. Early outreach to vulnerable communities may help prevent further escalation.';
  } else {
    action = 'No immediate action is required. Continued monitoring is advisable to detect any emerging trends before they become critical.';
  }

  return opening + breakdown + action;
}

async function captureChart(elementId, isDark) {
  const el = document.getElementById(elementId);
  if (!el) return null;
  try {
    const canvas = await html2canvas(el, {
      backgroundColor: isDark ? '#1E293B' : '#FFFFFF',
      scale: 1.5,
      logging: false,
      useCORS: true,
      allowTaint: true,
    });
    return canvas.toDataURL('image/png');
  } catch {
    return null;
  }
}

/* ─── header / footer helpers ──────────────────────────────── */

function drawHeader(doc, title, y) {
  doc.setFillColor(...BRAND_BLUE);
  doc.rect(MARGIN, y, CONTENT_W, 1.2, 'F');
  y += 4;
  doc.setFontSize(9);
  doc.setTextColor(...BRAND_BLUE);
  doc.setFont('helvetica', 'bold');
  doc.text(title.toUpperCase(), MARGIN, y);
  y += 1;
  doc.setDrawColor(...BRAND_BLUE);
  doc.setLineWidth(0.1);
  doc.line(MARGIN, y, MARGIN + CONTENT_W, y);
  return y + 5;
}

function drawFooter(doc) {
  const y = PAGE_H - 10;
  doc.setFontSize(7);
  doc.setTextColor(150, 150, 150);
  doc.setFont('helvetica', 'normal');
  doc.text(
    'Covenant University  |  Department of Computer Science and Information Systems  |  Developed by Oghenetejiri Ekpokpobe  |  2025',
    PAGE_W / 2,
    y,
    { align: 'center' }
  );
  doc.setDrawColor(200, 200, 200);
  doc.setLineWidth(0.2);
  doc.line(MARGIN, y - 4, MARGIN + CONTENT_W, y - 4);
}

function ensureSpace(doc, y, needed) {
  if (y + needed > PAGE_H - 16) {
    doc.addPage();
    drawFooter(doc);
    return MARGIN + 6;
  }
  return y;
}

/* ─── main export ──────────────────────────────────────────── */

export async function generateDashboardPDF({ summary, filters, isDark, recentTweets = [] }) {
  const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });

  // ── PAGE 1: Cover ──────────────────────────────────────────
  // Title block
  doc.setFillColor(...BRAND_BLUE);
  doc.rect(0, 0, PAGE_W, 38, 'F');
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(15);
  doc.setFont('helvetica', 'bold');
  doc.text('Food Crisis Sentiment Analysis Report', PAGE_W / 2, 16, { align: 'center' });
  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  doc.text('Lagos Twitter Data  —  AI-Powered Monitoring System', PAGE_W / 2, 24, { align: 'center' });
  doc.setFontSize(8);
  const now = new Date();
  doc.text(
    `Generated: ${now.toLocaleDateString('en-GB', { day: '2-digit', month: 'long', year: 'numeric' })}  at  ${now.toLocaleTimeString('en-GB')}`,
    PAGE_W / 2,
    31,
    { align: 'center' }
  );

  let y = 50;

  // ── Report parameters ──
  y = drawHeader(doc, 'Report Parameters', y);
  doc.setFontSize(9);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(40, 40, 40);

  const params = [
    ['Date Range', `Last ${filters?.days ?? 30} days`],
    ['Location Filter', filters?.location ?? 'All Lagos'],
    ['Model Used', modelLabel(filters?.model ?? 'bert')],
  ];
  params.forEach(([label, value]) => {
    doc.setFont('helvetica', 'bold');
    doc.text(`${label}:`, MARGIN, y);
    doc.setFont('helvetica', 'normal');
    doc.text(value, MARGIN + 38, y);
    y += 6;
  });

  y += 4;

  // ── Summary statistics ──
  y = drawHeader(doc, 'Summary Statistics', y);
  doc.setFontSize(9);

  const risk = riskLevel(summary?.negative_pct ?? 0);

  const stats = [
    { label: 'Total Tweets Analysed', value: (summary?.total_tweets ?? 0).toLocaleString(), color: [40, 40, 40] },
    { label: 'Negative Sentiment',    value: `${summary?.negative_pct ?? 0}%`,              color: NEGATIVE_R },
    { label: 'Neutral Sentiment',     value: `${summary?.neutral_pct ?? 0}%`,               color: NEUTRAL_R  },
    { label: 'Positive Sentiment',    value: `${summary?.positive_pct ?? 0}%`,              color: POSITIVE_R },
    { label: 'Current Crisis Risk',   value: risk.label,                                    color: risk.color },
  ];

  stats.forEach(({ label, value, color }) => {
    doc.setTextColor(80, 80, 80);
    doc.setFont('helvetica', 'normal');
    doc.text(label, MARGIN, y);
    doc.setTextColor(...color);
    doc.setFont('helvetica', 'bold');
    doc.text(value, MARGIN + CONTENT_W, y, { align: 'right' });
    doc.setDrawColor(220, 220, 220);
    doc.setLineWidth(0.1);
    doc.line(MARGIN, y + 1.5, MARGIN + CONTENT_W, y + 1.5);
    y += 8;
  });

  y += 6;

  // ── Plain English Summary ──
  y = drawHeader(doc, 'Situation Summary', y);
  const summaryText = plainEnglishSummary(summary, filters);
  const lines = doc.splitTextToSize(summaryText, CONTENT_W);
  doc.setFontSize(9);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(50, 50, 50);
  lines.forEach((line) => {
    y = ensureSpace(doc, y, 6);
    doc.text(line, MARGIN, y);
    y += 5.5;
  });

  drawFooter(doc);

  // ── PAGE 2: Charts ──────────────────────────────────────────
  const chartIds = [
    { id: 'pdf-chart-trend',    label: 'Sentiment Trend Over Time' },
    { id: 'pdf-chart-keywords', label: 'Top Keywords by Frequency' },
    { id: 'pdf-chart-volume',   label: 'Tweet Volume Over Time' },
  ];

  for (const { id, label } of chartIds) {
    const img = await captureChart(id, isDark);
    if (!img) continue;

    doc.addPage();
    drawFooter(doc);
    let cy = MARGIN;

    cy = drawHeader(doc, label, cy);
    const maxH = 110;
    doc.addImage(img, 'PNG', MARGIN, cy, CONTENT_W, maxH);
  }

  // ── PAGE N: Top 10 recent tweets ────────────────────────────
  if (recentTweets && recentTweets.length > 0) {
    doc.addPage();
    drawFooter(doc);
    let ty = MARGIN;
    ty = drawHeader(doc, 'Top 10 Most Recent Tweets', ty);

    const top10 = recentTweets.slice(0, 10);

    const colX = {
      tweet: MARGIN,
      label: MARGIN + CONTENT_W * 0.64,
      conf:  MARGIN + CONTENT_W * 0.82,
    };

    // Column headers
    doc.setFontSize(8);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(80, 80, 80);
    doc.setFillColor(240, 240, 240);
    doc.rect(MARGIN, ty - 4, CONTENT_W, 7, 'F');
    doc.text('Tweet',      colX.tweet, ty);
    doc.text('Sentiment',  colX.label, ty);
    doc.text('Confidence', colX.conf,  ty);
    ty += 5;

    top10.forEach((tweet, idx) => {
      const maxTweetW = CONTENT_W * 0.60;
      const tweetLines = doc.splitTextToSize(tweet.text || '', maxTweetW);
      const rowH = Math.max(tweetLines.length * 4.5 + 4, 10);

      ty = ensureSpace(doc, ty, rowH);

      // Alternating row bg
      if (idx % 2 === 0) {
        doc.setFillColor(248, 248, 248);
        doc.rect(MARGIN, ty - 3.5, CONTENT_W, rowH, 'F');
      }

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(7.5);
      doc.setTextColor(40, 40, 40);
      doc.text(tweetLines, colX.tweet, ty);

      const sentColor =
        tweet.label === 'Negative' ? NEGATIVE_R :
        tweet.label === 'Positive' ? POSITIVE_R :
        NEUTRAL_R;

      doc.setTextColor(...sentColor);
      doc.setFont('helvetica', 'bold');
      doc.text(tweet.label || '—', colX.label, ty);

      doc.setTextColor(60, 60, 60);
      doc.setFont('helvetica', 'normal');
      const conf = tweet.confidence != null
        ? `${Number(tweet.confidence).toFixed(1)}%`
        : '—';
      doc.text(conf, colX.conf, ty);

      ty += rowH;
    });
  }

  // Save
  const fileName = `food-crisis-report-${now.toISOString().slice(0, 10)}.pdf`;
  doc.save(fileName);
}
