import React from 'react';
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
} from 'chart.js';
import { useTheme } from '../contexts/ThemeContext';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const thresholdPlugin = {
  id: 'thresholdPlugin',
  afterDraw: (chart) => {
    const { ctx, chartArea: { left, right }, scales } = chart;
    const yScale = scales.y;
    const crisisY  = yScale.getPixelForValue(50);
    const warningY = yScale.getPixelForValue(35);

    const drawLine = (y, color) => {
      ctx.save();
      ctx.strokeStyle = color;
      ctx.setLineDash([6, 6]);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(left, y);
      ctx.lineTo(right, y);
      ctx.stroke();
      ctx.restore();
    };

    drawLine(crisisY,  '#C0392B');
    drawLine(warningY, '#E67E22');
  },
};

function SentimentTrendChart({ labels, negative, neutral, positive }) {
  const { isDark } = useTheme();

  const tickColor   = isDark ? '#94A3B8' : '#666666';
  const gridColor   = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
  const legendColor = isDark ? '#CBD5E1' : '#444444';
  const titleColor  = isDark ? '#CBD5E1' : '#444444';

  const data = {
    labels,
    datasets: [
      {
        label: 'Negative',
        data: negative,
        borderColor: '#C0392B',
        backgroundColor: 'rgba(192, 57, 43, 0.15)',
        tension: 0.4,
        fill: false,
      },
      {
        label: 'Neutral',
        data: neutral,
        borderColor: '#7F8C8D',
        backgroundColor: 'rgba(127, 140, 141, 0.15)',
        tension: 0.4,
        fill: false,
      },
      {
        label: 'Positive',
        data: positive,
        borderColor: '#27AE60',
        backgroundColor: 'rgba(39, 174, 96, 0.15)',
        tension: 0.4,
        fill: false,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
        labels: { color: legendColor },
      },
      title: {
        display: true,
        text: 'Sentiment Trend Over Time',
        color: titleColor,
      },
    },
    scales: {
      y: {
        min: 0,
        max: 100,
        ticks: {
          callback: (value) => `${value}%`,
          color: tickColor,
        },
        grid: { color: gridColor },
      },
      x: {
        ticks: {
          maxRotation: 0,
          minRotation: 0,
          color: tickColor,
        },
        grid: { color: gridColor },
      },
    },
  };

  return <Line data={data} options={options} plugins={[thresholdPlugin]} />;
}

export default SentimentTrendChart;
