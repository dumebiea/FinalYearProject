import React from 'react';

function LocationHeatmap({ locations }) {
  if (!locations || locations.length === 0) {
    return <div className="no-data">No location data available</div>;
  }

  const sortedLocations = [...locations].sort((a, b) => {
    const first = Number(a.negative_pct ?? 0);
    const second = Number(b.negative_pct ?? 0);
    return second - first;
  });

  const renderStatus = (pct) => {
    if (pct > 50) return '🔴 Crisis';
    if (pct > 35) return '⚠️ Warning';
    return '✅ Normal';
  };

  const getRowClass = (pct) => {
    if (pct > 50) return 'heatmap-row crisis';
    if (pct > 35) return 'heatmap-row warning';
    return 'heatmap-row normal';
  };

  return (
    <div className="location-heatmap">
      <table>
        <thead>
          <tr>
            <th>LGA Name</th>
            <th>Total Tweets</th>
            <th>Negative %</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {sortedLocations.map((location, index) => {
            const pct = Number(location.negative_pct ?? 0);
            return (
              <tr key={index} className={getRowClass(pct)}>
                <td>{location.name}</td>
                <td>{location.total}</td>
                <td>{pct}%</td>
                <td>{renderStatus(pct)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default LocationHeatmap;
