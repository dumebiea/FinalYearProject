import React from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { useTheme } from '../contexts/ThemeContext';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function KeywordBarChart({ keywords, frequencies }) {
  const { isDark } = useTheme();

  const tickColor  = isDark ? '#94A3B8' : '#666666';
  const gridColor  = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
  const titleColor = isDark ? '#CBD5E1' : '#444444';

  const data = {
    labels: keywords,
    datasets: [
      {
        label: 'Frequency',
        data: frequencies,
        backgroundColor: '#185FA5',
        borderColor: '#185FA5',
        borderWidth: 1,
      },
    ],
  };

  const options = {
    indexAxis: 'y',
    responsive: true,
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: 'Top Keywords by Frequency',
        color: titleColor,
      },
      tooltip: {
        callbacks: {
          label: (context) => `${context.parsed.x}`,
        },
      },
    },
    scales: {
      x: {
        beginAtZero: true,
        ticks: { color: tickColor },
        grid:  { color: gridColor },
      },
      y: {
        ticks: {
          autoSkip: false,
          color: tickColor,
        },
        grid: { color: gridColor },
      },
    },
  };

  return <Bar data={data} options={options} />;
}

export default KeywordBarChart;
