import React, { useState } from 'react';
import '../styles/HowToUse.css';

const SECTIONS = [
  {
    id: 'dashboard',
    icon: '📊',
    title: 'Dashboard',
    summary: 'Overview of real-time sentiment analysis results and live tweet activity.',
    items: [
      {
        heading: 'What the Dashboard Shows',
        body: 'The dashboard is your main control panel. At the top you will see four summary cards showing the total number of tweets analysed, the percentage of negative, neutral, and positive sentiment, and the current crisis risk level for Lagos.',
      },
      {
        heading: 'How to Read the Sentiment Charts',
        body: 'Below the summary cards are three charts. The Sentiment Trend chart shows how positive, neutral, and negative sentiment has changed day by day over the selected period. The Keyword Frequency chart shows the most common words found in negative tweets. The Tweet Activity chart shows how many tweets were collected each day.',
      },
      {
        heading: 'How to Use the Date Range Selector',
        body: 'Use the "Last N days" dropdown at the top of the dashboard to change the time period being analysed. You can select 7, 14, 30, or 60 days. The charts and summary cards will update automatically when you change this.',
      },
      {
        heading: 'How to Use the Model Selector',
        body: 'Use the model dropdown to choose which AI model classifies the live tweet stream. The options are BERT (most accurate), SVM, Logistic Regression, and Ensemble (combines all three models for a majority vote). BERT is selected by default.',
      },
      {
        heading: 'How to Read the Live Tweet Feed',
        body: 'The live feed at the bottom of the dashboard shows tweets as they are processed in real time. Each tweet displays the tweet text, its sentiment label (Negative, Neutral, or Positive), and the confidence score of the AI classification.',
      },
      {
        heading: 'How to Use Pause, Resume, and Reset',
        body: 'The Pause button stops the live tweet stream temporarily so you can read the current tweets without new ones pushing them off the screen. The Resume button restarts the stream. The Reset button clears all tweets from the live feed and starts fresh.',
      },
      {
        heading: 'How to Interpret the Crisis Alerts',
        body: 'If the percentage of negative tweets crosses a threshold, a coloured alert banner will appear at the top of the dashboard. A yellow Warning alert means negative sentiment is elevated (above 35%). A red Crisis alert means it has reached a critical level (above 50%) and requires immediate attention.',
      },
    ],
  },
  {
    id: 'classify',
    icon: '🐦',
    title: 'Classify Tweet',
    summary: 'Classify a single tweet\'s sentiment instantly using any AI model.',
    items: [
      {
        heading: 'How to Type or Paste a Tweet',
        body: 'Click inside the text box on the Classify Tweet page and either type a tweet directly or paste one you have copied. The text box accepts any length of text, but for best results keep it similar to a real tweet — a short sentence or paragraph about food, hunger, or food prices in Lagos.',
      },
      {
        heading: 'How to Select a Model',
        body: 'Use the model dropdown below the text box to choose which AI model will classify your tweet. BERT gives the most accurate results, SVM and Logistic Regression are faster alternatives, and Ensemble combines all three models and uses a majority vote for the final result.',
      },
      {
        heading: 'How to Read the Classification Result',
        body: 'After clicking the Classify button, the result panel will show the sentiment label: Negative, Neutral, or Positive. A short plain English message will also explain what the classification means in simple terms.',
      },
      {
        heading: 'What the Confidence Score Means',
        body: 'The confidence score (shown as a percentage) tells you how certain the AI model is about its classification. A score above 85% means the model is very confident. A score below 60% means the text was ambiguous and the result should be treated with caution.',
      },
      {
        heading: 'How to Use the Ensemble Option',
        body: 'When you select Ensemble from the model dropdown, the system runs all three models (BERT, SVM, and Logistic Regression) on your tweet and combines their votes. The result panel will show each individual model\'s prediction alongside the final combined verdict. This is the most reliable classification method.',
      },
    ],
  },
  {
    id: 'upload',
    icon: '📁',
    title: 'Upload Page',
    summary: 'Upload a CSV file to classify many tweets at once in a batch.',
    items: [
      {
        heading: 'How to Upload a CSV File',
        body: 'On the Upload Tweets page, click the "Choose File" or "Browse" button and select a CSV file from your computer. Once selected, the filename will appear and you can then click the Upload & Classify button to begin processing.',
      },
      {
        heading: 'What Format the CSV Should Be In',
        body: 'Your CSV file must contain a column named "text" (all lowercase) that holds the tweet text. Each row in the CSV is one tweet. The file can also contain other columns such as date or location — these will be ignored. Make sure the file is saved in UTF-8 encoding to avoid character errors.',
      },
      {
        heading: 'How to Select a Model for Batch Classification',
        body: 'Before uploading, use the model dropdown to choose which AI model will be used to classify all the tweets in your file. BERT is recommended for accuracy but may be slower on large files. SVM and Logistic Regression are faster alternatives.',
      },
      {
        heading: 'How to Read the Results Table',
        body: 'After processing, a table will appear showing each tweet alongside its predicted sentiment label (Negative, Neutral, or Positive) and the confidence score. You can scroll through the table to review all results. Negative tweets are highlighted in red, positive in green, and neutral in grey.',
      },
      {
        heading: 'How to Export the Results',
        body: 'Click the "Download Results" or "Export CSV" button below the results table to save the classified tweets as a new CSV file. This file will contain the original tweet text plus the predicted label and confidence score added as new columns.',
      },
    ],
  },
  {
    id: 'alerts',
    icon: '🚨',
    title: 'Alerts Page',
    summary: 'View crisis forecasts, active alerts, and a plain English risk summary.',
    items: [
      {
        heading: 'How to Read the Forecast Chart',
        body: 'The Alerts page shows a 7-day forecast chart that predicts how sentiment is expected to change over the coming week, based on current trends. The chart shows the predicted percentage of negative sentiment for each of the next seven days. A rising line means the situation is expected to worsen.',
      },
      {
        heading: 'How to Interpret the Crisis Risk Level',
        body: 'Below the forecast chart is a risk level indicator. It will show one of three levels — Low (green), Warning (yellow), or Crisis (red) — based on the predicted negative sentiment percentage. Low means the situation is stable. Warning means it is elevated. Crisis means immediate attention is recommended.',
      },
      {
        heading: 'How to Understand the Plain English Summary',
        body: 'Below the risk indicator is a short paragraph written in plain English that explains the current and predicted situation in everyday language. It summarises what the data means without technical jargon, making it easy for any reader to understand the sentiment outlook.',
      },
      {
        heading: 'What Triggers a Crisis Alert',
        body: 'A Warning alert is triggered automatically when the percentage of negative tweets in a day exceeds 35%. A Crisis alert is triggered when it exceeds 50%. These thresholds are checked once daily by the system. Once an alert is generated it appears in the alerts list with the date, location, and the exact negative sentiment percentage that triggered it.',
      },
    ],
  },
];

function HowToUse() {
  const [openSection, setOpenSection] = useState(null);

  const toggle = (id) => setOpenSection((prev) => (prev === id ? null : id));

  return (
    <div className="howto-page">
      <div className="howto-header">
        <h1 className="howto-title">How To Use This System</h1>
        <p className="howto-subtitle">
          A plain English guide to every part of the Food Crisis Sentiment Analysis System.
          Click a section to expand it.
        </p>
      </div>

      <div className="howto-sections">
        {SECTIONS.map((sec) => {
          const isOpen = openSection === sec.id;
          return (
            <div key={sec.id} className={`howto-section${isOpen ? ' howto-section--open' : ''}`}>
              {/* Section header (clickable) */}
              <button
                className="howto-section-header"
                onClick={() => toggle(sec.id)}
                aria-expanded={isOpen}
              >
                <div className="howto-section-left">
                  <span className="howto-section-icon">{sec.icon}</span>
                  <div className="howto-section-meta">
                    <span className="howto-section-title">{sec.title}</span>
                    <span className="howto-section-summary">
                      <span
                        className="howto-info-icon"
                        title={sec.summary}
                        aria-label={`About ${sec.title}`}
                      >
                        ℹ
                      </span>
                      {sec.summary}
                    </span>
                  </div>
                </div>
                <span className={`howto-chevron${isOpen ? ' howto-chevron--up' : ''}`}>›</span>
              </button>

              {/* Section body */}
              {isOpen && (
                <div className="howto-section-body">
                  {sec.items.map((item, idx) => (
                    <div key={idx} className="howto-item">
                      <h3 className="howto-item-heading">{item.heading}</h3>
                      <p className="howto-item-body">{item.body}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="howto-footer-note">
        <span className="howto-footer-icon">💡</span>
        For any issues or questions, contact the system administrator or refer to the project documentation.
      </div>
    </div>
  );
}

export default HowToUse;
