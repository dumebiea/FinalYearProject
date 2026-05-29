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

function TweetVolumeChart({ labels, volumes }) {
  const { isDark } = useTheme();

  const tickColor  = isDark ? '#94A3B8' : '#666666';
  const gridColor  = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
  const titleColor = isDark ? '#CBD5E1' : '#444444';

  const data = {
    labels,
    datasets: [
      {
        label: 'Tweet Volume',
        data: volumes,
        backgroundColor: 'rgba(24, 95, 165, 0.7)',
        borderColor: '#185FA5',
        borderWidth: 1,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: 'Tweet Volume Over Time',
        color: titleColor,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: { color: tickColor },
        grid:  { color: gridColor },
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

  return <Bar data={data} options={options} />;
}

export default TweetVolumeChart;
