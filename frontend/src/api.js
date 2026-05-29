import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:5000/api',
});

api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

export default api;

export const login = (email, password) =>
  api.post('/auth/login', { email, password });

export const getMe = () =>
  api.get('/auth/me');

export const getDashboardSummary = (days = 30, location = 'All Lagos') =>
  api.get('/dashboard/summary', { params: { days, location } });

export const getDashboardTrends = (days = 30, location = 'All Lagos') =>
  api.get('/dashboard/trends', { params: { days, location } });

export const getDashboardKeywords = (days = 7, sentiment = 'Negative') =>
  api.get('/dashboard/keywords', { params: { days, sentiment } });

export const getDashboardHeatmap = (days = 30) =>
  api.get('/dashboard/heatmap', { params: { days } });

export const getAlerts = (page = 1, limit = 20) =>
  api.get('/alerts', { params: { page, limit } });

export const getActiveAlerts = () =>
  api.get('/alerts/active');

export const acknowledgeAlert = (alertId) =>
  api.patch(`/alerts/${alertId}/acknowledge`);

export const predictTweet = (tweet, model = 'bert') =>
  api.post('/predict', { tweet, model });

export const uploadCSV = (formData, model = 'bert') => {
  formData.append('model', model);
  return api.post('/tweets/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const searchTweets = (keywords, location, model = 'bert') =>
  api.post('/tweets/search', { keywords, location, model });

export const getSystemHealth = () =>
  api.get('/health');

export const getStreamStatus = (model = null) =>
  api.get('/dashboard/stream-status', model ? { params: { model } } : {});
