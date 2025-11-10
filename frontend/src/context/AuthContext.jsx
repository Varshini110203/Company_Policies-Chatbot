import React, { createContext, useState, useContext, useEffect } from 'react';
import { authAPI, userAPI } from '../services/api';

const AuthContext = createContext();

export function useAuth() {
  return useContext(AuthContext);
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user is logged in on app start
    const token = localStorage.getItem('token');
    if (token) {
      getCurrentUser();
    } else {
      setLoading(false);
    }
  }, []);

  const getCurrentUser = async () => {
    try {
      const response = await userAPI.getCurrentUser();
      setUser(response.data);
    } catch (error) {
      console.error('Error getting current user:', error);
      localStorage.removeItem('token');
    } finally {
      setLoading(false);
    }
  };

  const login = async (email, password) => {
    try {
      const response = await authAPI.login(email, password);
      const { access_token } = response.data;
      
      localStorage.setItem('token', access_token);
      
      // Get user profile
      const userResponse = await userAPI.getCurrentUser();
      setUser(userResponse.data);
      
      return response.data;
    } catch (error) {
      throw error;
    }
  };

  const register = async (userData) => {
    try {
      const response = await authAPI.register(userData);
      return response.data;
    } catch (error) {
      throw error;
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    setUser(null);
  };

  const resetPassword = async (token, newPassword) => {
  try {
    const response = await authAPI.resetPassword(token, newPassword);
    return response.data;
  } catch (error) {
    throw error;
  }
};
  const value = {
    user,
    login,
    register,
    logout,
    resetPassword, // Add this line to include the function in context
    loading
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}