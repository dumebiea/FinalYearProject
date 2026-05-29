import React, { useState } from 'react';

function AlertBanner({ alerts, ready }) {
  const [dismissed, setDismissed] = useState(false);

  if (!ready || dismissed || !alerts || alerts.length === 0) {
    return null;
  }

  const crisisAlert = alerts.find((alert) => (alert.type || alert.alert_type) === 'crisis');
  const warningAlert = alerts.find((alert) => (alert.type || alert.alert_type) === 'warning');
  const activeAlert = crisisAlert || warningAlert;
  const isCrisis = Boolean(crisisAlert);

  if (!activeAlert) {
    return null;
  }

  const negativePct = (activeAlert.negative_pct ?? activeAlert.negative) || 0;
  const location = activeAlert.location || activeAlert.location_name || 'Lagos';

  return (
    <div className={`alert-banner ${isCrisis ? 'crisis' : 'warning'}`}>
      <div className="alert-text">
        <span className="alert-icon">{isCrisis ? '🚨' : '⚠️'}</span>
        <div>
          <strong>{isCrisis ? 'CRISIS ALERT' : 'WARNING'}</strong>
          <p>
            {isCrisis
              ? `${negativePct}% negative sentiment detected in ${location}`
              : `${negativePct}% negative sentiment — elevated food distress detected`}
          </p>
        </div>
      </div>
      <button className="alert-dismiss" onClick={() => setDismissed(true)}>
        Dismiss
      </button>
    </div>
  );
}

export default AlertBanner;
