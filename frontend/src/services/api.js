import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
});

// Add token to requests if available
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor to handle errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const authAPI = {
  login: async (email, password) => {
    const formData = new URLSearchParams();
    formData.append('username', email);
    formData.append('password', password);
    
    return api.post('/login', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
  },

  register: async (userData) => {
    return api.post('/register', userData);
  },

  resetPassword: async (token, newPassword) => {
    return api.post('/auth/reset-password', { 
      token, 
      new_password: newPassword 
    });
  },
};

export const userAPI = {
  getCurrentUser: () => api.get('/user/me'),
};

export const chatAPI = {
  sendMessage: (message, chatId = null) => 
    api.post('/chat/query', { message, chat_id: chatId }),
  
  getHistory: () => api.get('/chat/history'),
  
  deleteChat: (chatId) => api.delete(`/chat/${chatId}`),
  
  clearAllChats: () => api.delete('/chat/history'),
};

export default api;