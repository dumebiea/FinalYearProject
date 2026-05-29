import React from 'react';
import { Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
} from 'chart.js';
import { useTheme } from '../contexts/ThemeContext';

ChartJS.register(ArcElement, Tooltip, Legend);

const segmentLabelPlugin = {
  id: 'segmentLabelPlugin',
  afterDraw: (chart) => {
    const { ctx } = chart;
    chart.data.datasets.forEach((dataset, datasetIndex) => {
      const meta = chart.getDatasetMeta(datasetIndex);
      meta.data.forEach((element, index) => {
        const { x, y } = element.tooltipPosition();
        const value = dataset.data[index];
        const text = `${value}%`;

        ctx.save();
        ctx.fillStyle = '#ffffff';
        ctx.font = '600 12px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, x, y);
        ctx.restore();
      });
    });
  },
};

function SentimentPieChart({ negative, neutral, positive }) {
  const { isDark } = useTheme();

  const legendColor  = isDark ? '#CBD5E1' : '#444444';

  const data = {
    labels: ['Negative', 'Neutral', 'Positive'],
    datasets: [
      {
        data: [negative, neutral, positive],
        backgroundColor: ['#C0392B', '#7F8C8D', '#27AE60'],
        borderColor: isDark ? '#1E293B' : '#FFFFFF',
        borderWidth: 2,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'bottom',
        labels: { color: legendColor, padding: 16 },
      },
      tooltip: {
        callbacks: {
          label: (context) => `${context.label}: ${context.parsed}%`,
        },
      },
    },
  };

  return <Doughnut data={data} options={options} plugins={[segmentLabelPlugin]} />;
}

export default SentimentPieChart;
